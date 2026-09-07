//! `ignis-server` CLI flags mirroring the existing env-var config surface
//! one-to-one, plus `--help`/`-h` and `--version`/`-V` (GitHub #77,
//! `.scratch/spec-cli-config.md`).
//!
//! [`resolve`] is the one seam for this feature: pure (no `std::env`, no
//! filesystem, no process exit), so precedence and validation are covered by
//! fast unit tests instead of only end-to-end runs. `main` calls it once and
//! does nothing else config-related.

use std::path::PathBuf;

use crate::thinking::{self, ReasoningEffort};

/// The default loaded-model id (the v1 specialization: Qwen 3.8-27B —
/// `CONTEXT.md`).
pub const DEFAULT_MODEL: &str = "qwen3.8-27b";

/// The default bind address: localhost, port 8000 (OpenAI convention).
pub const DEFAULT_BIND: &str = "127.0.0.1:8000";

// The prefill-chunk and per-sequence-context defaults live in
// `ignis_runtime` (re-exported below), the same numbers `CudaLeafConfig`
// falls back to — one source of truth for what `ignis-server` runs with
// when the operator passes no flags, rather than two constants that have
// to be kept in sync by hand across the crate boundary.
pub use ignis_runtime::{DEFAULT_MAX_CONTEXT, DEFAULT_PREFILL_CHUNK, PREFILL_CHUNK_ALIGNMENT};

/// The fully-resolved config `main` needs to start the server — one field
/// per env var, each independently resolved as flag → env → default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub model: String,
    pub bind: String,
    pub artifact: Option<PathBuf>,
    pub telemetry: Option<PathBuf>,
    pub enable_thinking: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    /// The prefill chunk width, in tokens (a nonzero multiple of
    /// [`PREFILL_CHUNK_ALIGNMENT`]).
    pub prefill_chunk: u32,
    /// The maximum per-sequence context, in tokens (the largest prompt +
    /// generation budget a single request may reserve).
    pub max_context: u32,
    /// The paged-KV pool budget, in sequence-tokens: not independently
    /// configurable (spec `02-real-prefill.md` names only two engine-shape
    /// flags — the chunk width and the per-sequence cap). Derived from
    /// [`Config::max_context`] via [`ignis_runtime::kv_pool_tokens_for`],
    /// which is never smaller than it — a pool the cap cannot fit inside
    /// would admit a request the leaf can never allocate.
    pub kv_pool_tokens: u32,
}

/// What [`resolve`] produced: a runnable config, or a request to print
/// `--help`/`--version` text and exit before any loader/scheduler work runs.
/// `resolve` never prints or exits itself — that stays in `main`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigOutcome {
    Config(Config),
    Help(String),
    Version(String),
}

/// An unrecognized flag, a flag missing its required value, or a
/// thinking-parse failure surfaced from `thinking::parse_default_*` — the
/// message is suitable for `eprintln!("ignis-server: {err}")` before exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Resolve `args` (argv without the program name) and `env` (injected so
/// tests never touch the real process environment) into a [`ConfigOutcome`].
///
/// `--help`/`--version` short-circuit before any other flag is parsed or
/// validated — `ignis-server --help --nonsense` just prints help.
pub fn resolve(
    args: &[String],
    env: impl Fn(&str) -> Option<String>,
) -> Result<ConfigOutcome, ConfigError> {
    for arg in args {
        match arg.as_str() {
            "--help" | "-h" => return Ok(ConfigOutcome::Help(help_text())),
            "--version" | "-V" => return Ok(ConfigOutcome::Version(version_text())),
            _ => {}
        }
    }

    let mut model = None;
    let mut bind = None;
    let mut artifact = None;
    let mut telemetry = None;
    let mut enable_thinking = None;
    let mut reasoning_effort = None;
    let mut prefill_chunk = None;
    let mut max_context = None;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--model" | "-m" => model = Some(take_value(args, &mut i, flag)?),
            "--bind" | "-b" => bind = Some(take_value(args, &mut i, flag)?),
            "--artifact" | "-a" => artifact = Some(take_value(args, &mut i, flag)?),
            "--telemetry" | "-t" => telemetry = Some(take_value(args, &mut i, flag)?),
            "--enable-thinking" => enable_thinking = Some(take_value(args, &mut i, flag)?),
            "--reasoning-effort" => reasoning_effort = Some(take_value(args, &mut i, flag)?),
            "--prefill-chunk" => prefill_chunk = Some(take_value(args, &mut i, flag)?),
            "--max-context" => max_context = Some(take_value(args, &mut i, flag)?),
            other => return Err(ConfigError(format!("unrecognized flag `{other}`"))),
        }
        i += 1;
    }

    let model = model
        .or_else(|| env("IGNIS_MODEL"))
        .unwrap_or_else(|| DEFAULT_MODEL.to_owned());
    let bind = bind
        .or_else(|| env("IGNIS_BIND"))
        .unwrap_or_else(|| DEFAULT_BIND.to_owned());
    let artifact = non_empty(artifact.or_else(|| env("IGNIS_ARTIFACT"))).map(PathBuf::from);
    let telemetry = non_empty(telemetry.or_else(|| env("IGNIS_TELEMETRY"))).map(PathBuf::from);

    let enable_thinking_raw = enable_thinking
        .or_else(|| env("IGNIS_ENABLE_THINKING"))
        .unwrap_or_else(|| "true".to_owned());
    let enable_thinking =
        thinking::parse_default_enable_thinking(&enable_thinking_raw).map_err(ConfigError)?;

    let reasoning_effort_raw = reasoning_effort
        .or_else(|| env("IGNIS_REASONING_EFFORT"))
        .unwrap_or_default();
    let reasoning_effort =
        thinking::parse_default_reasoning_effort(&reasoning_effort_raw).map_err(ConfigError)?;

    // The engine-shape values (GitHub #87): resolved and validated here,
    // before `main` opens the artifact or touches the loader — an
    // unaligned chunk width is a usage error, never a failure discovered
    // after a ~19 GB weight upload.
    let prefill_chunk = resolve_prefill_chunk(prefill_chunk, &env)?;
    let max_context = resolve_max_context(max_context, &env)?;
    let kv_pool_tokens = ignis_runtime::kv_pool_tokens_for(max_context);

    Ok(ConfigOutcome::Config(Config {
        model,
        bind,
        artifact,
        telemetry,
        enable_thinking,
        reasoning_effort,
        prefill_chunk,
        max_context,
        kv_pool_tokens,
    }))
}

/// Parse a token-count value (`--prefill-chunk`, `--max-context`), naming
/// the flag and the offending text on failure.
fn parse_tokens(flag: &str, raw: &str) -> Result<u32, ConfigError> {
    raw.trim()
        .parse::<u32>()
        .map_err(|_| ConfigError(format!("`{flag}` expects a token count, got `{raw}`")))
}

/// `--prefill-chunk` / `IGNIS_PREFILL_CHUNK` / [`DEFAULT_PREFILL_CHUNK`].
fn resolve_prefill_chunk(
    flag: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<u32, ConfigError> {
    let Some(raw) = non_empty(flag.or_else(|| env("IGNIS_PREFILL_CHUNK"))) else {
        return Ok(DEFAULT_PREFILL_CHUNK);
    };
    let chunk = parse_tokens("--prefill-chunk", &raw)?;
    if chunk == 0 || chunk % PREFILL_CHUNK_ALIGNMENT != 0 {
        return Err(ConfigError(format!(
            "`--prefill-chunk` must be a nonzero multiple of {PREFILL_CHUNK_ALIGNMENT} tokens, got {chunk}"
        )));
    }
    Ok(chunk)
}

/// `--max-context` / `IGNIS_MAX_CONTEXT` / [`DEFAULT_MAX_CONTEXT`].
fn resolve_max_context(
    flag: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<u32, ConfigError> {
    let Some(raw) = non_empty(flag.or_else(|| env("IGNIS_MAX_CONTEXT"))) else {
        return Ok(DEFAULT_MAX_CONTEXT);
    };
    let context = parse_tokens("--max-context", &raw)?;
    if context == 0 {
        return Err(ConfigError(
            "`--max-context` must be a nonzero token count".to_owned(),
        ));
    }
    Ok(context)
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, ConfigError> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| ConfigError(format!("`{flag}` requires a value")))
}

fn version_text() -> String {
    format!("ignis-server {}", env!("CARGO_PKG_VERSION"))
}

fn help_text() -> String {
    format!(
        "ignis-server: the OpenAI-compatible HTTP entrypoint (localhost, no auth)\n\
         \n\
         USAGE:\n    ignis-server [OPTIONS]\n\
         \n\
         OPTIONS:\n\
         \x20   -m, --model <id>              env: IGNIS_MODEL         (default: {DEFAULT_MODEL})\n\
         \x20   -b, --bind <addr>             env: IGNIS_BIND          (default: {DEFAULT_BIND})\n\
         \x20   -a, --artifact <path>         env: IGNIS_ARTIFACT      (default: unset — placeholder template)\n\
         \x20   -t, --telemetry <path>        env: IGNIS_TELEMETRY     (default: unset — stdout)\n\
         \x20       --enable-thinking <bool>  env: IGNIS_ENABLE_THINKING   (default: true)\n\
         \x20       --reasoning-effort <val>  env: IGNIS_REASONING_EFFORT (default: unset — template default)\n\
         \x20       --prefill-chunk <tokens>  env: IGNIS_PREFILL_CHUNK  (default: {DEFAULT_PREFILL_CHUNK}; nonzero multiple of {PREFILL_CHUNK_ALIGNMENT})\n\
         \x20       --max-context <tokens>    env: IGNIS_MAX_CONTEXT    (default: {DEFAULT_MAX_CONTEXT}; max per-sequence prompt + generation)\n\
         \x20   -h, --help                    print this help and exit\n\
         \x20   -V, --version                 print the version and exit\n\
         \n\
         A flag overrides its env var, which overrides the built-in default."
    )
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

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn expect_config(outcome: ConfigOutcome) -> Config {
        match outcome {
            ConfigOutcome::Config(c) => c,
            other => panic!("expected Config, got {other:?}"),
        }
    }

    #[test]
    fn no_args_no_env_falls_back_to_defaults() {
        let config = expect_config(resolve(&[], no_env).expect("resolve"));
        assert_eq!(config.model, DEFAULT_MODEL);
        assert_eq!(config.bind, DEFAULT_BIND);
        assert_eq!(config.artifact, None);
        assert_eq!(config.telemetry, None);
        assert!(config.enable_thinking);
        assert_eq!(config.reasoning_effort, None);
        assert_eq!(config.prefill_chunk, DEFAULT_PREFILL_CHUNK);
        assert_eq!(config.max_context, DEFAULT_MAX_CONTEXT);
        assert_eq!(config.kv_pool_tokens, ignis_runtime::kv_pool_tokens_for(DEFAULT_MAX_CONTEXT));
    }

    #[test]
    fn env_only_wins_over_defaults() {
        let env = env_map(&[
            ("IGNIS_MODEL", "custom-model"),
            ("IGNIS_BIND", "0.0.0.0:9000"),
            ("IGNIS_ARTIFACT", "/path/to.ninfer"),
            ("IGNIS_TELEMETRY", "/tmp/telemetry.jsonl"),
            ("IGNIS_ENABLE_THINKING", "false"),
            ("IGNIS_REASONING_EFFORT", "low"),
        ]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.model, "custom-model");
        assert_eq!(config.bind, "0.0.0.0:9000");
        assert_eq!(config.artifact, Some(PathBuf::from("/path/to.ninfer")));
        assert_eq!(config.telemetry, Some(PathBuf::from("/tmp/telemetry.jsonl")));
        assert!(!config.enable_thinking);
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::Low));
    }

    #[test]
    fn flag_only_wins_over_defaults() {
        let a = args(&[
            "--model", "flag-model",
            "--bind", "0.0.0.0:1234",
            "--artifact", "/flag/artifact.ninfer",
            "--telemetry", "/flag/telemetry.jsonl",
            "--enable-thinking", "false",
            "--reasoning-effort", "high",
        ]);
        let config = expect_config(resolve(&a, no_env).expect("resolve"));
        assert_eq!(config.model, "flag-model");
        assert_eq!(config.bind, "0.0.0.0:1234");
        assert_eq!(config.artifact, Some(PathBuf::from("/flag/artifact.ninfer")));
        assert_eq!(config.telemetry, Some(PathBuf::from("/flag/telemetry.jsonl")));
        assert!(!config.enable_thinking);
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn short_aliases_behave_like_their_long_form() {
        let a = args(&["-m", "m", "-b", "b", "-a", "a", "-t", "t"]);
        let config = expect_config(resolve(&a, no_env).expect("resolve"));
        assert_eq!(config.model, "m");
        assert_eq!(config.bind, "b");
        assert_eq!(config.artifact, Some(PathBuf::from("a")));
        assert_eq!(config.telemetry, Some(PathBuf::from("t")));
    }

    #[test]
    fn a_flag_wins_over_a_matching_env_var_per_field_independently() {
        let env = env_map(&[
            ("IGNIS_BIND", "0.0.0.0:9000"),
            ("IGNIS_ARTIFACT", "/env/artifact.ninfer"),
        ]);
        let a = args(&["--bind", "0.0.0.0:1234"]);
        let config = expect_config(resolve(&a, env).expect("resolve"));
        assert_eq!(config.bind, "0.0.0.0:1234", "flag must win over env");
        assert_eq!(
            config.artifact,
            Some(PathBuf::from("/env/artifact.ninfer")),
            "env must still apply to a field the flag didn't touch"
        );
    }

    #[test]
    fn an_unrecognized_flag_is_a_config_error() {
        let a = args(&["--nope"]);
        let err = resolve(&a, no_env).expect_err("must reject");
        assert!(err.0.contains("--nope"), "{err}");
    }

    #[test]
    fn a_flag_missing_its_value_is_a_config_error() {
        let a = args(&["--bind"]);
        let err = resolve(&a, no_env).expect_err("must reject");
        assert!(err.0.contains("--bind"), "{err}");
    }

    #[test]
    fn an_invalid_enable_thinking_flag_matches_the_env_var_error_message() {
        let flag_err = resolve(&args(&["--enable-thinking", "nope"]), no_env)
            .expect_err("must reject");
        let env_err = thinking::parse_default_enable_thinking("nope").unwrap_err();
        assert_eq!(flag_err.0, env_err);
    }

    #[test]
    fn an_invalid_reasoning_effort_flag_matches_the_env_var_error_message() {
        let flag_err = resolve(&args(&["--reasoning-effort", "nonsense"]), no_env)
            .expect_err("must reject");
        let env_err = thinking::parse_default_reasoning_effort("nonsense").unwrap_err();
        assert_eq!(flag_err.0, env_err);
    }

    #[test]
    fn help_short_circuits_before_other_flags_are_validated() {
        let outcome = resolve(&args(&["--help", "--nonsense"]), no_env).expect("resolve");
        assert!(matches!(outcome, ConfigOutcome::Help(_)));
    }

    #[test]
    fn help_short_circuits_even_where_it_would_otherwise_be_consumed_as_a_value() {
        // `--bind` normally requires a following value; `--help` still wins
        // rather than being swallowed as that value, matching "short-circuits
        // before further parsing" for any position in argv.
        let outcome = resolve(&args(&["--bind", "--help"]), no_env).expect("resolve");
        assert!(matches!(outcome, ConfigOutcome::Help(_)));
    }

    #[test]
    fn help_alias_short_circuits_too() {
        let outcome = resolve(&args(&["-h"]), no_env).expect("resolve");
        assert!(matches!(outcome, ConfigOutcome::Help(_)));
    }

    #[test]
    fn version_short_circuits_before_other_flags_are_validated() {
        let outcome = resolve(&args(&["--version", "--nonsense"]), no_env).expect("resolve");
        assert!(matches!(outcome, ConfigOutcome::Version(_)));
    }

    #[test]
    fn version_alias_short_circuits_too() {
        let outcome = resolve(&args(&["-V"]), no_env).expect("resolve");
        assert!(matches!(outcome, ConfigOutcome::Version(_)));
    }

    // ── the engine-shape flags (GitHub #87) ──────────────────────────────

    #[test]
    fn the_default_context_admits_a_32k_prompt_plus_a_generation_budget() {
        // G2's largest cell is a 32,768-token prompt; the default cap must
        // admit it *and* leave room to generate, without editing code.
        let config = expect_config(resolve(&[], no_env).expect("resolve"));
        assert!(
            config.max_context > 32_768,
            "the default per-sequence context ({}) must admit a 32K prompt plus a generation budget",
            config.max_context
        );
        // The pool the leaf builds must be able to hold one such sequence.
        assert!(config.kv_pool_tokens >= config.max_context);
    }

    #[test]
    fn the_engine_shape_env_vars_win_over_the_defaults() {
        let env = env_map(&[("IGNIS_PREFILL_CHUNK", "2048"), ("IGNIS_MAX_CONTEXT", "16384")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.prefill_chunk, 2048);
        assert_eq!(config.max_context, 16_384);
    }

    #[test]
    fn the_engine_shape_flags_win_over_their_env_vars() {
        let env = env_map(&[("IGNIS_PREFILL_CHUNK", "2048"), ("IGNIS_MAX_CONTEXT", "16384")]);
        let a = args(&["--prefill-chunk", "128", "--max-context", "8192"]);
        let config = expect_config(resolve(&a, env).expect("resolve"));
        assert_eq!(config.prefill_chunk, 128, "flag must win over env");
        assert_eq!(config.max_context, 8_192, "flag must win over env");
    }

    #[test]
    fn an_unaligned_prefill_chunk_is_a_usage_error() {
        // The alignment rule is the reference's own; an unaligned width is
        // rejected before any loader work, not at the first long prompt.
        let err = resolve(&args(&["--prefill-chunk", "1000"]), no_env).expect_err("must reject");
        assert!(err.0.contains("128"), "the message must name the rule: {err}");
        assert!(err.0.contains("1000"), "the message must name the value: {err}");
    }

    #[test]
    fn a_zero_prefill_chunk_is_a_usage_error() {
        let err = resolve(&args(&["--prefill-chunk", "0"]), no_env).expect_err("must reject");
        assert!(err.0.contains("nonzero"), "{err}");
    }

    #[test]
    fn a_non_numeric_prefill_chunk_is_a_usage_error() {
        let err = resolve(&args(&["--prefill-chunk", "wide"]), no_env).expect_err("must reject");
        assert!(err.0.contains("--prefill-chunk"), "{err}");
    }

    #[test]
    fn an_invalid_prefill_chunk_env_var_is_a_usage_error_too() {
        // Same rule whichever way the value arrived (the env var is not a
        // back door around the validation).
        let env = env_map(&[("IGNIS_PREFILL_CHUNK", "300")]);
        let err = resolve(&[], env).expect_err("must reject");
        assert!(err.0.contains("128"), "{err}");
    }

    #[test]
    fn a_zero_max_context_is_a_usage_error() {
        let err = resolve(&args(&["--max-context", "0"]), no_env).expect_err("must reject");
        assert!(err.0.contains("--max-context"), "{err}");
    }

    #[test]
    fn the_pool_always_grows_to_hold_the_configured_cap() {
        // The pool is not independently configurable (spec
        // `02-real-prefill.md` names only `--prefill-chunk` and
        // `--max-context`); a cap above the default pool raises the pool
        // with it, automatically, so admission can never promise a context
        // the pool cannot hold.
        let a = args(&["--max-context", "200000"]);
        let config = expect_config(resolve(&a, no_env).expect("resolve"));
        assert_eq!(config.max_context, 200_000);
        assert!(config.kv_pool_tokens >= 200_000);
    }

    #[test]
    fn help_lists_the_engine_shape_flags() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        for flag in ["--prefill-chunk", "--max-context"] {
            assert!(text.contains(flag), "help must document {flag}:\n{text}");
        }
    }
}
