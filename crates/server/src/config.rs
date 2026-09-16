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

/// Where `--metrics` serves Prometheus when `--metrics-bind` names nowhere
/// else (GitHub #89, ADR 0017): localhost, on the port OpenTelemetry's
/// Prometheus exporter uses — its own listener, never the API's.
pub const DEFAULT_METRICS_BIND: &str = "127.0.0.1:9464";

/// The default non-streaming completion timeout, in seconds (GitHub #95) —
/// unchanged from the value `Server::new` hardcoded before this flag
/// existed.
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u32 = 30;

/// The upper bound `--request-timeout`/`IGNIS_REQUEST_TIMEOUT` accepts: a
/// ceiling against a fat-fingered value, not a real operating point — a
/// healthy request legitimately runs for minutes at a large `max_tokens`,
/// never hours.
pub const MAX_REQUEST_TIMEOUT_SECS: u32 = 3600;

// The prefill-chunk and per-sequence-context defaults live in
// `ignis_runtime` (re-exported below), the same numbers `CudaLeafConfig`
// falls back to — one source of truth for what `ignis-server` runs with
// when the operator passes no flags, rather than two constants that have
// to be kept in sync by hand across the crate boundary.
pub use ignis_runtime::{DEFAULT_MAX_CONTEXT, DEFAULT_PREFILL_CHUNK, PREFILL_CHUNK_ALIGNMENT};

pub use ignis_core::KvFormat;
pub use ignis_core::{MAX_DRAFT_TOKENS, Speculation, SpeculativeBackend};
pub use ignis_core::{DEFAULT_VISION_MAX_TOKENS, VISION_MAX_TOKENS_LIMIT, Vision};

/// The fully-resolved config `main` needs to start the server — one field
/// per env var, each independently resolved as flag → env → default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub model: String,
    pub bind: String,
    pub artifact: Option<PathBuf>,
    pub enable_thinking: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    /// The prefill chunk width, in tokens (a nonzero multiple of
    /// [`PREFILL_CHUNK_ALIGNMENT`]).
    pub prefill_chunk: u32,
    /// The maximum per-sequence context, in tokens (the largest prompt +
    /// generation budget a single request may reserve).
    pub max_context: u32,
    /// The KV storage format this load runs on (ADR 0022, GitHub #122),
    /// fixed for the life of the load.
    pub kv_format: KvFormat,
    /// The paged-KV pool budget, in **bytes**. Never in tokens: what the
    /// budget is worth in tokens is derived from [`Config::kv_format`] and
    /// reported at load. When the operator names none, this is
    /// [`ignis_runtime::auto_kv_pool_bytes`] for the resolved format and
    /// context — never smaller than one full context, since a pool the
    /// per-sequence cap cannot fit inside would admit a request the leaf
    /// can never allocate.
    pub kv_pool_bytes: u64,
    /// The KV-RAM host tier's budget, in bytes (P4-07, GitHub #125): pinned
    /// host memory for evicted (suspended) request snapshots. `0` disables
    /// the tier (admission refuses instead of evicting once the resident
    /// lanes are full). A byte budget, not a page or lane count, because a
    /// snapshot's fixed GDN floor (~145 MiB) is paid regardless of prompt
    /// length.
    pub host_pool_bytes: u64,
    /// Cross-request state reuse (`--prompt-reuse`, GitHub #186, ADR 0029).
    /// On by default. Off means a request captures no prompt checkpoint and
    /// claims none, so a cold bench measures a cold engine and a correctness
    /// oracle prefills every prompt it is given.
    pub prompt_reuse: bool,
    /// The retained checkpoint pool's device budget, in bytes
    /// (`--retained-pool-bytes`, GitHub #186). `0` retains nothing. A byte
    /// budget for the reason [`Config::host_pool_bytes`] is one: an image is
    /// dominated by a fixed state floor a short conversation pays exactly as
    /// a long one does. Unset derives a default from the VRAM left after
    /// load; this is what the operator asked for, `None` meaning "derive it".
    pub retained_pool_bytes: Option<u64>,
    /// Speculative decoding, chosen at load (`--spec`/`--draft-tokens`, P5-02
    /// GitHub #150). `None` loads nothing of the drafter.
    pub speculation: Option<Speculation>,
    /// Vision, chosen at load (`--vision`/`--vision-max-tokens`, GitHub #177).
    /// `None` binds and reserves nothing of the vision tower.
    pub vision: Option<Vision>,
    /// Media acquisition (`--media-allow-private-network`,
    /// `--media-cache-mib`, GitHub #179). Only nameable with vision on.
    pub media: MediaOptions,
    /// How long a non-streaming request waits for its completion before the
    /// handler gives up with a `504` (GitHub #95). In `[1, MAX_REQUEST_TIMEOUT_SECS]`.
    pub request_timeout_secs: u32,
    /// Serve the Playground under `/ui/` (`--ui`, GitHub #163, ADR 0026).
    /// Flag-only: no env var, no alias.
    pub ui: bool,
    /// The metrics listener's address when `--metrics` is on (GitHub #89,
    /// ADR 0017): `--metrics-bind`, else [`DEFAULT_METRICS_BIND`]. `None` =
    /// metrics off. Flag-only: no env var, no alias, no config-file key.
    pub metrics: Option<String>,
    /// The key every `/v1` request must present as `Authorization: Bearer
    /// <key>` (`--api-key` / `IGNIS_API_KEY`). `None` (the default) keeps
    /// the API open, as it has always been on localhost.
    pub api_key: Option<ApiKeySetting>,
    /// How the server is exposed beyond its bind address (`--expose` /
    /// `IGNIS_EXPOSE`, ADR 0028). `Some` always comes with an API key:
    /// without one, [`resolve`] sets `api_key` to [`ApiKeySetting::Generate`].
    pub expose: Option<Expose>,
}

pub use crate::expose::Expose;

/// `--media-cache-mib`'s default (the reference's 1 GiB).
pub const DEFAULT_MEDIA_CACHE_MIB: u32 = 1024;

/// The largest `--media-cache-mib` accepted: a ceiling against a
/// fat-fingered value (64 GiB of host memory for prepared patches).
pub const MEDIA_CACHE_MIB_LIMIT: u32 = 64 * 1024;

/// How image parts are acquired (GitHub #179).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaOptions {
    /// Fetch image URLs that resolve to private, loopback, link-local,
    /// multicast or CGNAT addresses. Off by default, so an `--expose`d
    /// server cannot be used to probe the operator's LAN.
    pub allow_private_network: bool,
    /// Host memory for prepared image patches kept for reuse, in bytes
    /// (0 retains nothing).
    pub cache_bytes: u64,
}

impl Default for MediaOptions {
    fn default() -> Self {
        Self { allow_private_network: false, cache_bytes: (DEFAULT_MEDIA_CACHE_MIB as u64) << 20 }
    }
}

/// What `--api-key` asked for: a key the operator chose, or `auto` — one
/// `main` generates at start and prints, the only time a key is printed.
/// Resolved here, generated in `main`, so [`resolve`] stays pure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiKeySetting {
    Fixed(ApiKey),
    Generate,
}

/// An API key. Its `Debug` never prints the value, so a `Config` dumped
/// into a log line or a test failure does not leak the secret.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// A fresh key from the OS random source: `sk-ignis-` and 256 random
    /// bits in hex.
    pub fn generate() -> Result<Self, getrandom::Error> {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes)?;
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        Ok(Self(format!("sk-ignis-{hex}")))
    }

    /// The key itself — for the one place that has to show it.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether `presented` is this key. Compares every byte whatever the
    /// first mismatch, so the time taken does not reveal a correct prefix.
    pub fn matches(&self, presented: &str) -> bool {
        let (a, b) = (self.0.as_bytes(), presented.as_bytes());
        a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey([REDACTED])")
    }
}

/// `--kv-host-pool-bytes` / `IGNIS_KV_HOST_POOL_BYTES`'s default (P4-07,
/// GitHub #125): comfortably holds several full-context snapshots (each
/// ~528 MB per ADR 0024's estimate) without an operator having to reason
/// about the format's per-snapshot cost just to start the server.
pub const DEFAULT_HOST_POOL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// `--prompt-reuse`'s default (GitHub #186, ADR 0029): on. Cross-request
/// reuse is the owner's workload — an agent's tool loop re-sends its whole
/// history every iteration — so it is what the engine does unless asked not
/// to.
pub const DEFAULT_PROMPT_REUSE: bool = true;

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
    let mut enable_thinking = None;
    let mut reasoning_effort = None;
    let mut prefill_chunk = None;
    let mut max_context = None;
    let mut kv_format = None;
    let mut kv_pool_bytes = None;
    let mut host_pool_bytes = None;
    let mut prompt_reuse = None;
    let mut retained_pool_bytes = None;
    let mut request_timeout = None;
    let mut spec = None;
    let mut draft_tokens = None;
    let mut vision = false;
    let mut vision_max_tokens = None;
    let mut media_allow_private_network = false;
    let mut media_cache_mib = None;
    let mut ui = false;
    let mut metrics_on = false;
    let mut metrics_bind = None;
    let mut api_key = None;
    let mut expose = None;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--model" | "-m" => model = Some(take_value(args, &mut i, flag)?),
            "--bind" | "-b" => bind = Some(take_value(args, &mut i, flag)?),
            "--artifact" | "-a" => artifact = Some(take_value(args, &mut i, flag)?),
            "--enable-thinking" => enable_thinking = Some(take_value(args, &mut i, flag)?),
            "--reasoning-effort" => reasoning_effort = Some(take_value(args, &mut i, flag)?),
            "--prefill-chunk" => prefill_chunk = Some(take_value(args, &mut i, flag)?),
            "--max-context" => max_context = Some(take_value(args, &mut i, flag)?),
            "--kv-format" => kv_format = Some(take_value(args, &mut i, flag)?),
            "--kv-pool-bytes" => kv_pool_bytes = Some(take_value(args, &mut i, flag)?),
            "--kv-host-pool-bytes" => host_pool_bytes = Some(take_value(args, &mut i, flag)?),
            "--prompt-reuse" => prompt_reuse = Some(take_value(args, &mut i, flag)?),
            "--retained-pool-bytes" => {
                retained_pool_bytes = Some(take_value(args, &mut i, flag)?)
            }
            "--request-timeout" => request_timeout = Some(take_value(args, &mut i, flag)?),
            "--spec" => spec = Some(take_value(args, &mut i, flag)?),
            "--draft-tokens" => draft_tokens = Some(take_value(args, &mut i, flag)?),
            "--vision" => vision = true,
            "--vision-max-tokens" => vision_max_tokens = Some(take_value(args, &mut i, flag)?),
            "--media-allow-private-network" => media_allow_private_network = true,
            "--media-cache-mib" => media_cache_mib = Some(take_value(args, &mut i, flag)?),
            "--ui" => ui = true,
            "--metrics" => metrics_on = true,
            "--metrics-bind" => metrics_bind = Some(take_value(args, &mut i, flag)?),
            "--api-key" => api_key = Some(take_value(args, &mut i, flag)?),
            "--expose" => expose = Some(take_value(args, &mut i, flag)?),
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
    // The format is resolved before the budget, because what a budget is
    // worth in tokens — and so what the auto default has to be — depends on
    // it (GitHub #122).
    let kv_format = resolve_kv_format(kv_format, &env)?;
    let kv_pool_bytes = resolve_kv_pool_bytes(kv_pool_bytes, &env, kv_format, max_context)?;
    let host_pool_bytes = resolve_host_pool_bytes(host_pool_bytes, &env)?;
    let prompt_reuse = resolve_prompt_reuse(prompt_reuse, &env)?;
    let retained_pool_bytes =
        resolve_retained_pool_bytes(retained_pool_bytes, &env, prompt_reuse)?;
    let request_timeout_secs = resolve_request_timeout_secs(request_timeout, &env)?;
    let speculation = resolve_speculation(spec, draft_tokens, &env)?;
    let vision = resolve_vision(vision, vision_max_tokens, &env)?;
    // GitHub #178: DFlash2's drafter does not follow a multimodal prompt yet,
    // so the two load options are refused together here, before any load
    // work, as the reference refuses `--spec dflash` with `--vision`.
    if vision.is_some() && speculation.is_some_and(|s| s.backend() == SpeculativeBackend::Dflash2) {
        return Err(ConfigError(
            "`--vision` cannot be combined with `--spec dflash2` yet (speculative decoding does not follow image prompts)"
                .to_owned(),
        ));
    }
    let media = resolve_media(vision.is_some(), media_allow_private_network, media_cache_mib, &env)?;
    // `--metrics` (GitHub #89, ADR 0017) opens its own listener; naming its
    // address without turning metrics on is refused rather than ignored, and
    // it can never share the API's.
    let metrics = match (metrics_on, metrics_bind) {
        (false, None) => None,
        (false, Some(_)) => {
            return Err(ConfigError(
                "`--metrics-bind` requires `--metrics` (metrics are off without it)".to_owned(),
            ));
        }
        (true, metrics_bind) => Some(metrics_bind.unwrap_or_else(|| DEFAULT_METRICS_BIND.to_owned())),
    };
    if metrics.as_deref() == Some(bind.as_str()) {
        return Err(ConfigError(format!(
            "`--metrics-bind {bind}` is the API's `--bind`: metrics need their own listener"
        )));
    }
    let api_key = non_empty(api_key.or_else(|| env("IGNIS_API_KEY"))).map(|key| match key.as_str() {
        "auto" => ApiKeySetting::Generate,
        _ => ApiKeySetting::Fixed(ApiKey(key)),
    });
    let expose = non_empty(expose.or_else(|| env("IGNIS_EXPOSE")))
        .map(|raw| Expose::parse(&raw).map_err(|e| ConfigError(format!("`--expose`: {e}"))))
        .transpose()?;
    // An exposed API is never open: the operator's key if they named one,
    // otherwise the same generated key `--api-key auto` gives.
    let api_key = match (&expose, api_key) {
        (Some(_), None) => Some(ApiKeySetting::Generate),
        (_, api_key) => api_key,
    };

    Ok(ConfigOutcome::Config(Config {
        model,
        bind,
        artifact,
        enable_thinking,
        reasoning_effort,
        prefill_chunk,
        max_context,
        kv_format,
        kv_pool_bytes,
        host_pool_bytes,
        prompt_reuse,
        retained_pool_bytes,
        speculation,
        vision,
        media,
        request_timeout_secs,
        ui,
        metrics,
        api_key,
        expose,
    }))
}

/// `--vision` / `IGNIS_VISION` and `--vision-max-tokens` /
/// `IGNIS_VISION_MAX_TOKENS` (GitHub #177). Vision is off unless asked for;
/// an envelope with vision off has nothing to size, so naming one alone is
/// refused rather than ignored. With vision on, the envelope defaults to
/// [`DEFAULT_VISION_MAX_TOKENS`].
fn resolve_vision(
    flag: bool,
    max_tokens: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<Option<Vision>, ConfigError> {
    let on = if flag {
        true
    } else {
        match non_empty(env("IGNIS_VISION")) {
            None => false,
            Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "on" => true,
                "0" | "false" | "off" => false,
                _ => {
                    return Err(ConfigError(format!(
                        "`IGNIS_VISION` must be true or false, got `{raw}`"
                    )));
                }
            },
        }
    };
    let max_tokens = max_tokens.or_else(|| non_empty(env("IGNIS_VISION_MAX_TOKENS")));
    if !on {
        return match max_tokens {
            Some(raw) => Err(ConfigError(format!(
                "`--vision-max-tokens {raw}` requires `--vision` (vision is off without it)"
            ))),
            None => Ok(None),
        };
    }
    let Some(raw) = max_tokens else {
        return Ok(Some(Vision::default()));
    };
    let out_of_range = || {
        ConfigError(format!(
            "`--vision-max-tokens` must be in 1..={VISION_MAX_TOKENS_LIMIT}, got `{raw}`"
        ))
    };
    let n = raw.trim().parse::<u32>().map_err(|_| out_of_range())?;
    Vision::new(n).map(Some).map_err(|_| out_of_range())
}

/// `--media-allow-private-network` / `IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK` and
/// `--media-cache-mib` / `IGNIS_MEDIA_CACHE_MIB` (GitHub #179). Without
/// vision there is no media to acquire, so naming either is refused rather
/// than ignored, as `--vision-max-tokens` is.
fn resolve_media(
    vision: bool,
    allow_private_network_flag: bool,
    cache_mib: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<MediaOptions, ConfigError> {
    let allow_private_network = if allow_private_network_flag {
        Some(true)
    } else {
        match non_empty(env("IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK")) {
            None => None,
            Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "on" => Some(true),
                "0" | "false" | "off" => Some(false),
                _ => {
                    return Err(ConfigError(format!(
                        "`IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK` must be true or false, got `{raw}`"
                    )));
                }
            },
        }
    };
    let cache_mib = cache_mib.or_else(|| non_empty(env("IGNIS_MEDIA_CACHE_MIB")));
    if !vision {
        if allow_private_network == Some(true) {
            return Err(ConfigError(
                "`--media-allow-private-network` requires `--vision` (vision is off without it)".to_owned(),
            ));
        }
        if let Some(raw) = cache_mib {
            return Err(ConfigError(format!(
                "`--media-cache-mib {raw}` requires `--vision` (vision is off without it)"
            )));
        }
        return Ok(MediaOptions::default());
    }
    let cache_mib = match cache_mib {
        None => DEFAULT_MEDIA_CACHE_MIB,
        Some(raw) => raw
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|&mib| mib <= MEDIA_CACHE_MIB_LIMIT)
            .ok_or_else(|| {
                ConfigError(format!(
                    "`--media-cache-mib` must be in 0..={MEDIA_CACHE_MIB_LIMIT}, got `{raw}`"
                ))
            })?,
    };
    Ok(MediaOptions {
        allow_private_network: allow_private_network.unwrap_or(false),
        cache_bytes: (cache_mib as u64) << 20,
    })
}

/// `--spec` / `IGNIS_SPEC` and `--draft-tokens` / `IGNIS_DRAFT_TOKENS`
/// (P5-02, GitHub #150). Absent `--spec` means off, and then a draft window
/// has nothing to size, so naming one alone is refused rather than ignored.
/// With `--spec`, the window is required — there is no default window to
/// guess — and anything outside `1..MAX_DRAFT_TOKENS` is refused naming the
/// range.
fn resolve_speculation(
    spec: Option<String>,
    draft_tokens: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<Option<Speculation>, ConfigError> {
    let spec = non_empty(spec.or_else(|| env("IGNIS_SPEC")));
    let draft_tokens = non_empty(draft_tokens.or_else(|| env("IGNIS_DRAFT_TOKENS")));
    let Some(spec) = spec else {
        return match draft_tokens {
            Some(raw) => Err(ConfigError(format!(
                "`--draft-tokens {raw}` requires `--spec` (speculation is off without it)"
            ))),
            None => Ok(None),
        };
    };
    let backend =
        SpeculativeBackend::parse(&spec).map_err(|e| ConfigError(format!("`--spec`: {e}")))?;
    let Some(raw) = draft_tokens else {
        return Err(ConfigError(format!(
            "`--spec {}` requires `--draft-tokens N` (N in 1..{MAX_DRAFT_TOKENS})",
            backend.as_str()
        )));
    };
    let out_of_range =
        || ConfigError(format!("`--draft-tokens` must be in 1..{MAX_DRAFT_TOKENS}, got `{raw}`"));
    let n = raw.trim().parse::<u32>().map_err(|_| out_of_range())?;
    Speculation::new(backend, n).map(Some).map_err(|_| out_of_range())
}

/// Parse a `u32` count for `flag`, naming the flag, `unit`, and the
/// offending text on failure.
fn parse_count(flag: &str, unit: &str, raw: &str) -> Result<u32, ConfigError> {
    raw.trim()
        .parse::<u32>()
        .map_err(|_| ConfigError(format!("`{flag}` expects a {unit}, got `{raw}`")))
}

/// A token-count value (`--prefill-chunk`, `--max-context`).
fn parse_tokens(flag: &str, raw: &str) -> Result<u32, ConfigError> {
    parse_count(flag, "token count", raw)
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

/// `--kv-format` / `IGNIS_KV_FORMAT` / [`KvFormat::default`]
/// (`hq-e8-2b`, the serving default since GitHub #123; `bf16` is the
/// retained oracle format an operator asks for by name).
fn resolve_kv_format(
    flag: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<KvFormat, ConfigError> {
    let Some(raw) = non_empty(flag.or_else(|| env("IGNIS_KV_FORMAT"))) else {
        return Ok(KvFormat::default());
    };
    KvFormat::parse(&raw).map_err(|e| ConfigError(format!("`--kv-format`: {e}")))
}

/// A byte-count value (`--kv-pool-bytes`), with the size suffixes an
/// operator actually types: a bare count, or one followed by `K`/`M`/`G`
/// (case-insensitive, binary — `4G` is 4 GiB), optionally spelled `KiB`,
/// `MiB`, `GiB` or `KB`/`MB`/`GB`. A pool budget is naturally a number of
/// gibibytes, and making the operator write 4294967296 invites the typo
/// that silently starts a server with a tenth of the pool it meant.
fn parse_bytes(flag: &str, raw: &str) -> Result<u64, ConfigError> {
    let text = raw.trim();
    let bad = || ConfigError(format!("`{flag}` expects a byte count, got `{raw}`"));
    let digits_end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let (digits, suffix) = text.split_at(digits_end);
    if digits.is_empty() {
        return Err(bad());
    }
    let multiplier: u64 = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        "g" | "gb" | "gib" => 1024 * 1024 * 1024,
        _ => return Err(bad()),
    };
    digits
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(multiplier))
        .ok_or_else(bad)
}

/// `--kv-pool-bytes` / `IGNIS_KV_POOL_BYTES` / the auto default for the
/// resolved format and context ([`ignis_runtime::auto_kv_pool_bytes`]).
///
/// An explicit budget is *not* raised to fit the context: if the operator
/// names one too small, that is a usage error caught here, before any
/// loader work — the auto default is what "big enough by construction"
/// means, and silently overriding an explicit number would make the flag a
/// suggestion.
fn resolve_kv_pool_bytes(
    flag: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
    format: KvFormat,
    max_context: u32,
) -> Result<u64, ConfigError> {
    let Some(raw) = non_empty(flag.or_else(|| env("IGNIS_KV_POOL_BYTES"))) else {
        return Ok(ignis_runtime::auto_kv_pool_bytes(format, max_context));
    };
    let bytes = parse_bytes("--kv-pool-bytes", &raw)?;
    ignis_core::plan_kv_pool_for_context(
        format,
        ignis_core::KvGeometry::qwen38_27b(),
        bytes,
        max_context,
    )
    .map_err(|e| ConfigError(format!("`--kv-pool-bytes`: {e}")))?;
    Ok(bytes)
}

/// `--prompt-reuse on|off` / `IGNIS_PROMPT_REUSE` / [`DEFAULT_PROMPT_REUSE`]
/// (GitHub #186, ADR 0029). A value, not a bare switch, because the useful
/// direction is *off* — a cold bench or a correctness oracle turning
/// something on-by-default back off — and a bare `--prompt-reuse` could only
/// ever ask for the default.
fn resolve_prompt_reuse(
    flag: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<bool, ConfigError> {
    let Some(raw) = non_empty(flag.or_else(|| env("IGNIS_PROMPT_REUSE"))) else {
        return Ok(DEFAULT_PROMPT_REUSE);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" => Ok(true),
        "0" | "false" | "off" => Ok(false),
        _ => Err(ConfigError(format!(
            "`--prompt-reuse` must be on or off, got `{raw}`"
        ))),
    }
}

/// `--retained-pool-bytes` / `IGNIS_RETAINED_POOL_BYTES` (GitHub #186).
/// Unset is `None`: the budget is then derived from the VRAM left after the
/// model lands, which only the loader can know.
///
/// Naming a budget with `--prompt-reuse off` is refused rather than ignored,
/// the house rule every other sub-flag follows: a pool that will never hold
/// anything is not what the operator meant to size.
fn resolve_retained_pool_bytes(
    flag: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
    prompt_reuse: bool,
) -> Result<Option<u64>, ConfigError> {
    let Some(raw) = non_empty(flag.or_else(|| env("IGNIS_RETAINED_POOL_BYTES"))) else {
        return Ok(None);
    };
    if !prompt_reuse {
        return Err(ConfigError(format!(
            "`--retained-pool-bytes {raw}` requires `--prompt-reuse on` (nothing is retained without it)"
        )));
    }
    parse_bytes("--retained-pool-bytes", &raw).map(Some)
}

/// `--kv-host-pool-bytes` / `IGNIS_KV_HOST_POOL_BYTES` / [`DEFAULT_HOST_POOL_BYTES`]
/// (P4-07, GitHub #125). `0` is a legal, explicit choice — it disables the
/// host tier (admission refuses instead of evicting) — so it is accepted
/// rather than treated as "unset" the way an empty string is.
fn resolve_host_pool_bytes(
    flag: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<u64, ConfigError> {
    let Some(raw) = non_empty(flag.or_else(|| env("IGNIS_KV_HOST_POOL_BYTES"))) else {
        return Ok(DEFAULT_HOST_POOL_BYTES);
    };
    parse_bytes("--kv-host-pool-bytes", &raw)
}

/// `--request-timeout` / `IGNIS_REQUEST_TIMEOUT` / [`DEFAULT_REQUEST_TIMEOUT_SECS`].
fn resolve_request_timeout_secs(
    flag: Option<String>,
    env: &impl Fn(&str) -> Option<String>,
) -> Result<u32, ConfigError> {
    let Some(raw) = non_empty(flag.or_else(|| env("IGNIS_REQUEST_TIMEOUT"))) else {
        return Ok(DEFAULT_REQUEST_TIMEOUT_SECS);
    };
    let secs = parse_count("--request-timeout", "second count", &raw)?;
    if secs == 0 {
        return Err(ConfigError(
            "`--request-timeout` must be a nonzero second count".to_owned(),
        ));
    }
    if secs > MAX_REQUEST_TIMEOUT_SECS {
        return Err(ConfigError(format!(
            "`--request-timeout` must be at most {MAX_REQUEST_TIMEOUT_SECS} seconds, got {secs}"
        )));
    }
    Ok(secs)
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
    let default_kv_format = KvFormat::default().as_str();
    let default_kv_pool_gib = ignis_core::DEFAULT_KV_POOL_BYTES / (1024 * 1024 * 1024);
    let default_host_pool_gib = DEFAULT_HOST_POOL_BYTES / (1024 * 1024 * 1024);
    format!(
        "ignis-server: the OpenAI-compatible HTTP entrypoint\n\
         \n\
         USAGE:\n    ignis-server [OPTIONS]\n\
         \n\
         OPTIONS:\n\
         \x20   -m, --model <id>              env: IGNIS_MODEL         (default: {DEFAULT_MODEL})\n\
         \x20   -b, --bind <addr>             env: IGNIS_BIND          (default: {DEFAULT_BIND})\n\
         \x20   -a, --artifact <path>         env: IGNIS_ARTIFACT      (default: unset — placeholder template)\n\
         \x20       --enable-thinking <bool>  env: IGNIS_ENABLE_THINKING   (default: true)\n\
         \x20       --reasoning-effort <val>  env: IGNIS_REASONING_EFFORT (default: unset — template default)\n\
         \x20       --prefill-chunk <tokens>  env: IGNIS_PREFILL_CHUNK  (default: {DEFAULT_PREFILL_CHUNK}; nonzero multiple of {PREFILL_CHUNK_ALIGNMENT})\n\
         \x20       --max-context <tokens>    env: IGNIS_MAX_CONTEXT    (default: {DEFAULT_MAX_CONTEXT}; max per-sequence prompt + generation)\n\
         \x20       --kv-format <fmt>         env: IGNIS_KV_FORMAT      (default: {default_kv_format}; bf16 or hq-e8-2b)\n\
         \x20       --kv-pool-bytes <bytes>   env: IGNIS_KV_POOL_BYTES  (default: auto, {default_kv_pool_gib} GiB; accepts a K/M/G suffix)\n\
         \x20       --kv-host-pool-bytes <b>  env: IGNIS_KV_HOST_POOL_BYTES (default: {default_host_pool_gib} GiB; 0 disables the host KV-RAM tier)\n\
         \x20       --prompt-reuse <on|off>   env: IGNIS_PROMPT_REUSE   (default: on; off = no prompt checkpoint is captured or reused)\n\
         \x20       --retained-pool-bytes <b> env: IGNIS_RETAINED_POOL_BYTES (default: derived from the VRAM left after load; needs --prompt-reuse on; accepts a K/M/G suffix)\n\
         \x20       --request-timeout <secs>  env: IGNIS_REQUEST_TIMEOUT (default: {DEFAULT_REQUEST_TIMEOUT_SECS}; max {MAX_REQUEST_TIMEOUT_SECS})\n\
         \x20       --spec <backend>          env: IGNIS_SPEC           (default: unset — no speculation; dflash2)\n\
         \x20       --draft-tokens <n>        env: IGNIS_DRAFT_TOKENS   (required with --spec; 1..{MAX_DRAFT_TOKENS})\n\
         \x20       --vision                  env: IGNIS_VISION         (default: off; load the vision tower and reserve its workspace)\n\
         \x20       --vision-max-tokens <n>   env: IGNIS_VISION_MAX_TOKENS (default: {DEFAULT_VISION_MAX_TOKENS} with --vision; merged vision tokens per request, 1..={VISION_MAX_TOKENS_LIMIT})\n\
         \x20       --media-allow-private-network env: IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK (default: off; needs --vision; fetch image URLs on private, loopback and link-local addresses)\n\
         \x20       --media-cache-mib <n>     env: IGNIS_MEDIA_CACHE_MIB (default: {DEFAULT_MEDIA_CACHE_MIB} with --vision; prepared images kept for reuse, 0 disables, max {MEDIA_CACHE_MIB_LIMIT})\n\
         \x20       --ui                      serve the Playground at /ui/ (default: off; flag only)\n\
         \x20       --metrics                 serve Prometheus metrics on their own listener, and at /ui/metrics with --ui (default: off; flag only)\n\
         \x20       --metrics-bind <addr>     the metrics listener (default: {DEFAULT_METRICS_BIND}; flag only; needs --metrics; no API key, never exposed)\n\
         \x20       --api-key <key>           env: IGNIS_API_KEY        (default: unset — /v1 needs no key; set = Authorization: Bearer <key>; auto = generate one and print it)\n\
         \x20       --expose <mode>           env: IGNIS_EXPOSE         (default: unset — reachable at --bind only; cloudflare-quick = public https://*.trycloudflare.com URL, printed at start; always requires an API key, auto when none is set)\n\
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
        assert!(config.enable_thinking);
        assert_eq!(config.reasoning_effort, None);
        assert_eq!(config.prefill_chunk, DEFAULT_PREFILL_CHUNK);
        assert_eq!(config.max_context, DEFAULT_MAX_CONTEXT);
        assert_eq!(config.kv_format, KvFormat::HqE8_2b);
        assert_eq!(
            config.kv_pool_bytes,
            ignis_runtime::auto_kv_pool_bytes(KvFormat::HqE8_2b, DEFAULT_MAX_CONTEXT)
        );
        assert_eq!(config.host_pool_bytes, DEFAULT_HOST_POOL_BYTES);
        assert_eq!(config.speculation, None);
        assert_eq!(config.request_timeout_secs, DEFAULT_REQUEST_TIMEOUT_SECS);
    }

    #[test]
    fn env_only_wins_over_defaults() {
        let env = env_map(&[
            ("IGNIS_MODEL", "custom-model"),
            ("IGNIS_BIND", "0.0.0.0:9000"),
            ("IGNIS_ARTIFACT", "/path/to.ninfer"),
            ("IGNIS_ENABLE_THINKING", "false"),
            ("IGNIS_REASONING_EFFORT", "low"),
        ]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.model, "custom-model");
        assert_eq!(config.bind, "0.0.0.0:9000");
        assert_eq!(config.artifact, Some(PathBuf::from("/path/to.ninfer")));
        assert!(!config.enable_thinking);
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::Low));
    }

    #[test]
    fn flag_only_wins_over_defaults() {
        let a = args(&[
            "--model", "flag-model",
            "--bind", "0.0.0.0:1234",
            "--artifact", "/flag/artifact.ninfer",
            "--enable-thinking", "false",
            "--reasoning-effort", "high",
        ]);
        let config = expect_config(resolve(&a, no_env).expect("resolve"));
        assert_eq!(config.model, "flag-model");
        assert_eq!(config.bind, "0.0.0.0:1234");
        assert_eq!(config.artifact, Some(PathBuf::from("/flag/artifact.ninfer")));
        assert!(!config.enable_thinking);
        assert_eq!(config.reasoning_effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn short_aliases_behave_like_their_long_form() {
        let a = args(&["-m", "m", "-b", "b", "-a", "a"]);
        let config = expect_config(resolve(&a, no_env).expect("resolve"));
        assert_eq!(config.model, "m");
        assert_eq!(config.bind, "b");
        assert_eq!(config.artifact, Some(PathBuf::from("a")));
    }

    #[test]
    fn the_retired_telemetry_sink_flag_is_refused_and_its_env_var_ignored() {
        // ADR 0025: the interval counters are a log event now, so there is
        // no separate sink to point anywhere. A leftover `--telemetry` in a
        // launch script must fail loudly rather than be silently dropped.
        for flag in ["--telemetry", "-t"] {
            let err = resolve(&args(&[flag, "/tmp/telemetry.jsonl"]), no_env)
                .expect_err("a retired flag must be rejected");
            assert!(err.0.contains(flag), "{err}");
        }
        let env = env_map(&[("IGNIS_TELEMETRY", "/tmp/telemetry.jsonl")]);
        assert_eq!(
            resolve(&[], env).expect("resolve"),
            resolve(&[], no_env).expect("resolve"),
            "IGNIS_TELEMETRY no longer changes the resolved config"
        );

        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(!text.contains("telemetry"), "help must not document it:\n{text}");
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
        let plan = ignis_core::plan_kv_pool(
            config.kv_format,
            ignis_core::KvGeometry::qwen38_27b(),
            config.kv_pool_bytes,
        );
        assert!(plan.token_capacity >= u64::from(config.max_context));
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
    fn the_auto_pool_budget_always_grows_to_hold_the_configured_cap() {
        // A cap above the default budget raises the budget with it,
        // automatically, so admission can never promise a context the pool
        // cannot hold. The cap at which that happens is format-dependent --
        // 4 GiB buys 65,536 BF16 tokens and 465,984 hq ones -- so each arm
        // names a context past its own format's default capacity rather than
        // one number that only stresses whichever format happens to be the
        // default (GitHub #123 made that hq).
        for (format, context) in [("bf16", 200_000u32), ("hq-e8-2b", 600_000)] {
            let a = args(&["--kv-format", format, "--max-context", &context.to_string()]);
            let config = expect_config(resolve(&a, no_env).expect("resolve"));
            assert_eq!(config.max_context, context);
            assert!(
                config.kv_pool_bytes > ignis_core::DEFAULT_KV_POOL_BYTES,
                "{format} at {context}: the budget did not grow past the default"
            );
            let plan = ignis_core::plan_kv_pool(
                config.kv_format,
                ignis_core::KvGeometry::qwen38_27b(),
                config.kv_pool_bytes,
            );
            assert!(plan.token_capacity >= u64::from(context), "{format} at {context}");
        }
    }

    #[test]
    fn help_lists_the_engine_shape_flags() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        for flag in [
            "--prefill-chunk",
            "--max-context",
            "--kv-format",
            "--kv-pool-bytes",
        ] {
            assert!(text.contains(flag), "help must document {flag}:\n{text}");
        }
        assert!(text.contains("hq-e8-2b"), "help must name both formats:\n{text}");
    }

    // ── the KV format and pool budget (GitHub #122) ──────────────────────

    #[test]
    fn the_kv_format_flag_wins_over_the_env_var_and_the_default() {
        let env = env_map(&[("IGNIS_KV_FORMAT", "bf16")]);
        let config = expect_config(resolve(&args(&["--kv-format", "hq-e8-2b"]), env).expect("resolve"));
        assert_eq!(config.kv_format, KvFormat::HqE8_2b);

        // Both halves name the format the default is *not*, so neither can
        // pass by agreeing with it (GitHub #123 made the default hq-e8-2b).
        let env = env_map(&[("IGNIS_KV_FORMAT", "bf16")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.kv_format, KvFormat::Bf16);
    }

    #[test]
    fn an_unknown_kv_format_is_a_usage_error() {
        let err = resolve(&args(&["--kv-format", "fp8"]), no_env).expect_err("unknown format");
        assert!(err.0.contains("--kv-format") && err.0.contains("fp8"), "{}", err.0);
    }

    #[test]
    fn the_same_default_budget_buys_more_tokens_under_hq() {
        // The format is a real option: one budget, two capacities. This is
        // the whole reason the pool is described in bytes.
        let geometry = ignis_core::KvGeometry::qwen38_27b();
        let bf16 = expect_config(resolve(&args(&["--kv-format", "bf16"]), no_env).expect("resolve"));
        let hq = expect_config(resolve(&args(&["--kv-format", "hq-e8-2b"]), no_env).expect("resolve"));
        assert_eq!(bf16.kv_pool_bytes, hq.kv_pool_bytes);
        let bf16_capacity =
            ignis_core::plan_kv_pool(bf16.kv_format, geometry, bf16.kv_pool_bytes).token_capacity;
        let hq_capacity =
            ignis_core::plan_kv_pool(hq.kv_format, geometry, hq.kv_pool_bytes).token_capacity;
        assert!(hq_capacity > bf16_capacity * 7, "{hq_capacity} vs {bf16_capacity}");
        // And it clears the standard target profile: 8 lanes x 40,960.
        assert!(hq_capacity >= 8 * 40_960);
    }

    #[test]
    fn an_explicit_pool_budget_overrides_the_auto_default() {
        let config =
            expect_config(resolve(&args(&["--kv-pool-bytes", "8G"]), no_env).expect("resolve"));
        assert_eq!(config.kv_pool_bytes, 8 * 1024 * 1024 * 1024);

        let env = env_map(&[("IGNIS_KV_POOL_BYTES", "6144MiB")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.kv_pool_bytes, 6144 * 1024 * 1024);

        // A bare count is still a byte count.
        let config = expect_config(
            resolve(&args(&["--kv-pool-bytes", "4294967296"]), no_env).expect("resolve"),
        );
        assert_eq!(config.kv_pool_bytes, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn a_pool_budget_too_small_for_the_context_is_refused_before_any_loader_work() {
        // 1 MiB cannot hold a 40,960-token sequence in either format. The
        // message has to name the budget, the format and the capacity it
        // bought, so the operator can see which of the three to change --
        // and it names whichever format is actually in force, which is why
        // both are asked here.
        for format in ["bf16", "hq-e8-2b"] {
            let err = resolve(&args(&["--kv-pool-bytes", "1M", "--kv-format", format]), no_env)
                .expect_err("a budget this small");
            assert!(err.0.contains("--kv-pool-bytes"), "{}", err.0);
            assert!(err.0.contains(format), "{}", err.0);
            assert!(err.0.contains("40960"), "{}", err.0);
        }
    }

    #[test]
    fn a_budget_big_enough_only_under_hq_is_accepted_only_under_hq() {
        // 512 MiB holds a 40,960-token sequence under hq (378 MB) and not
        // under BF16 (2.5 GiB) — the format decides whether the load starts.
        let too_small_for_bf16 = args(&["--kv-pool-bytes", "512M", "--kv-format", "bf16"]);
        assert!(resolve(&too_small_for_bf16, no_env).is_err());

        let under_hq = args(&["--kv-pool-bytes", "512M", "--kv-format", "hq-e8-2b"]);
        let config = expect_config(resolve(&under_hq, no_env).expect("resolve"));
        assert_eq!(config.kv_pool_bytes, 512 * 1024 * 1024);
    }

    #[test]
    fn a_malformed_pool_budget_is_a_usage_error() {
        for raw in ["", "4 GiB please", "-1", "4TB", "G"] {
            let a = args(&["--kv-pool-bytes", raw]);
            match resolve(&a, no_env) {
                // An empty value falls through to the auto default, the
                // same as every other flag here (`non_empty`).
                Ok(_) if raw.is_empty() => {}
                Ok(_) => panic!("`{raw}` must not parse as a byte count"),
                Err(err) => assert!(err.0.contains("--kv-pool-bytes"), "{}", err.0),
            }
        }
    }

    // ── the KV-RAM host tier byte budget (P4-07, GitHub #125) ────────────

    #[test]
    fn an_explicit_host_pool_budget_overrides_the_default() {
        let config = expect_config(
            resolve(&args(&["--kv-host-pool-bytes", "512M"]), no_env).expect("resolve"),
        );
        assert_eq!(config.host_pool_bytes, 512 * 1024 * 1024);

        let env = env_map(&[("IGNIS_KV_HOST_POOL_BYTES", "1G")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.host_pool_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn the_host_pool_budget_flag_wins_over_its_env_var() {
        let env = env_map(&[("IGNIS_KV_HOST_POOL_BYTES", "1G")]);
        let a = args(&["--kv-host-pool-bytes", "256M"]);
        let config = expect_config(resolve(&a, env).expect("resolve"));
        assert_eq!(config.host_pool_bytes, 256 * 1024 * 1024, "flag must win over env");
    }

    // ── GitHub #186: cross-request state reuse (ADR 0029) ──────────────

    #[test]
    fn prompt_reuse_is_on_unless_the_operator_turns_it_off() {
        let config = expect_config(resolve(&[], no_env).expect("resolve"));
        assert!(config.prompt_reuse, "on by default (ADR 0029)");
        assert_eq!(
            config.retained_pool_bytes, None,
            "no budget named: derived from the VRAM left after load"
        );

        let config =
            expect_config(resolve(&args(&["--prompt-reuse", "off"]), no_env).expect("resolve"));
        assert!(!config.prompt_reuse);

        let env = env_map(&[("IGNIS_PROMPT_REUSE", "off")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert!(!config.prompt_reuse, "the env var turns it off too");

        // A flag wins over its env var, in both directions.
        let env = env_map(&[("IGNIS_PROMPT_REUSE", "off")]);
        let a = args(&["--prompt-reuse", "on"]);
        let config = expect_config(resolve(&a, env).expect("resolve"));
        assert!(config.prompt_reuse, "flag must win over env");
    }

    #[test]
    fn a_malformed_prompt_reuse_value_is_a_usage_error() {
        let err = resolve(&args(&["--prompt-reuse", "maybe"]), no_env)
            .expect_err("`maybe` is neither on nor off");
        assert!(err.0.contains("--prompt-reuse"), "{}", err.0);
        assert!(err.0.contains("maybe"), "names the value: {}", err.0);
    }

    #[test]
    fn an_explicit_retained_pool_budget_overrides_the_derived_one() {
        let config = expect_config(
            resolve(&args(&["--retained-pool-bytes", "512M"]), no_env).expect("resolve"),
        );
        assert_eq!(config.retained_pool_bytes, Some(512 * 1024 * 1024));

        let env = env_map(&[("IGNIS_RETAINED_POOL_BYTES", "1G")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.retained_pool_bytes, Some(1024 * 1024 * 1024));

        let env = env_map(&[("IGNIS_RETAINED_POOL_BYTES", "1G")]);
        let a = args(&["--retained-pool-bytes", "256M"]);
        let config = expect_config(resolve(&a, env).expect("resolve"));
        assert_eq!(
            config.retained_pool_bytes,
            Some(256 * 1024 * 1024),
            "flag must win over env"
        );
    }

    #[test]
    fn a_retained_pool_budget_without_reuse_is_refused_not_ignored() {
        // The house rule every other sub-flag follows: sizing a pool that
        // will never hold anything is not what the operator meant.
        let a = args(&["--prompt-reuse", "off", "--retained-pool-bytes", "512M"]);
        let err = resolve(&a, no_env).expect_err("a budget with reuse off");
        assert!(err.0.contains("--retained-pool-bytes"), "{}", err.0);
        assert!(err.0.contains("--prompt-reuse"), "names what it needs: {}", err.0);
    }

    #[test]
    fn a_zero_host_pool_budget_is_accepted_and_disables_the_tier() {
        // Unlike an empty string (falls through to the default), `0` is an
        // explicit, legal operator choice: no host tier at all.
        let config =
            expect_config(resolve(&args(&["--kv-host-pool-bytes", "0"]), no_env).expect("resolve"));
        assert_eq!(config.host_pool_bytes, 0);
    }

    #[test]
    fn a_malformed_host_pool_budget_is_a_usage_error() {
        let err = resolve(&args(&["--kv-host-pool-bytes", "not-a-size"]), no_env)
            .expect_err("must reject");
        assert!(err.0.contains("--kv-host-pool-bytes"), "{}", err.0);
    }

    #[test]
    fn help_lists_the_host_pool_budget_flag() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(
            text.contains("--kv-host-pool-bytes"),
            "help must document --kv-host-pool-bytes:\n{text}"
        );
    }

    // ── the request timeout (GitHub #95) ─────────────────────────────────

    #[test]
    fn the_request_timeout_env_var_wins_over_the_default() {
        let env = env_map(&[("IGNIS_REQUEST_TIMEOUT", "90")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.request_timeout_secs, 90);
    }

    #[test]
    fn the_request_timeout_flag_wins_over_its_env_var() {
        let env = env_map(&[("IGNIS_REQUEST_TIMEOUT", "90")]);
        let a = args(&["--request-timeout", "45"]);
        let config = expect_config(resolve(&a, env).expect("resolve"));
        assert_eq!(config.request_timeout_secs, 45, "flag must win over env");
    }

    #[test]
    fn a_zero_request_timeout_is_a_usage_error() {
        let err = resolve(&args(&["--request-timeout", "0"]), no_env).expect_err("must reject");
        assert!(err.0.contains("nonzero"), "{err}");
    }

    #[test]
    fn a_non_numeric_request_timeout_is_a_usage_error() {
        let err =
            resolve(&args(&["--request-timeout", "soon"]), no_env).expect_err("must reject");
        assert!(err.0.contains("--request-timeout"), "{err}");
    }

    #[test]
    fn a_request_timeout_above_the_ceiling_is_a_usage_error() {
        let err = resolve(&args(&["--request-timeout", "3601"]), no_env).expect_err("must reject");
        assert!(err.0.contains("3600"), "the message must name the ceiling: {err}");
        assert!(err.0.contains("3601"), "the message must name the value: {err}");
    }

    #[test]
    fn an_invalid_request_timeout_env_var_is_a_usage_error_too() {
        let env = env_map(&[("IGNIS_REQUEST_TIMEOUT", "0")]);
        let err = resolve(&[], env).expect_err("must reject");
        assert!(err.0.contains("nonzero"), "{err}");
    }

    #[test]
    fn help_lists_the_request_timeout_flag() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(text.contains("--request-timeout"), "help must document --request-timeout:\n{text}");
    }

    // ── speculation as a load option (P5-02, GitHub #150) ────────────────

    #[test]
    fn spec_dflash2_with_a_window_in_range_parses() {
        for n in 1..=7u32 {
            let a = args(&["--spec", "dflash2", "--draft-tokens", &n.to_string()]);
            let config = expect_config(resolve(&a, no_env).expect("resolve"));
            assert_eq!(
                config.speculation,
                Some(Speculation::new(SpeculativeBackend::Dflash2, n).unwrap())
            );
        }
    }

    #[test]
    fn a_draft_window_outside_1_to_7_is_refused_naming_the_range() {
        for raw in ["0", "8", "15", "-1", "seven"] {
            let a = args(&["--spec", "dflash2", "--draft-tokens", raw]);
            let err = resolve(&a, no_env).expect_err("out of range");
            assert!(err.0.contains("--draft-tokens"), "{}", err.0);
            assert!(err.0.contains("1..7"), "{}", err.0);
            assert!(err.0.contains(raw), "{}", err.0);
        }
    }

    #[test]
    fn an_unknown_speculative_backend_is_refused_naming_dflash2() {
        let a = args(&["--spec", "mtp", "--draft-tokens", "3"]);
        let err = resolve(&a, no_env).expect_err("unknown backend");
        assert!(err.0.contains("--spec") && err.0.contains("mtp"), "{}", err.0);
        assert!(err.0.contains("dflash2"), "{}", err.0);
    }

    #[test]
    fn spec_without_a_draft_window_is_refused() {
        let err = resolve(&args(&["--spec", "dflash2"]), no_env).expect_err("no window");
        assert!(err.0.contains("--draft-tokens") && err.0.contains("1..7"), "{}", err.0);
    }

    #[test]
    fn a_draft_window_without_spec_is_refused_rather_than_ignored() {
        let err = resolve(&args(&["--draft-tokens", "7"]), no_env).expect_err("no backend");
        assert!(err.0.contains("--spec"), "{}", err.0);
    }

    #[test]
    fn the_speculation_env_vars_apply_and_the_flags_win_over_them() {
        let env = env_map(&[("IGNIS_SPEC", "dflash2"), ("IGNIS_DRAFT_TOKENS", "3")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.speculation.map(|s| s.draft_tokens()), Some(3));

        let env = env_map(&[("IGNIS_SPEC", "dflash2"), ("IGNIS_DRAFT_TOKENS", "3")]);
        let config =
            expect_config(resolve(&args(&["--draft-tokens", "7"]), env).expect("resolve"));
        assert_eq!(config.speculation.map(|s| s.draft_tokens()), Some(7), "flag must win over env");
    }

    // ── vision as a load option (GitHub #177) ─────────────────────────────

    #[test]
    fn vision_is_off_by_default() {
        let config = expect_config(resolve(&[], no_env).expect("resolve"));
        assert_eq!(config.vision, None);
    }

    #[test]
    fn the_vision_flag_loads_the_default_envelope() {
        let config = expect_config(resolve(&args(&["--vision"]), no_env).expect("resolve"));
        assert_eq!(config.vision, Some(Vision::default()));
        assert_eq!(config.vision.unwrap().max_tokens(), DEFAULT_VISION_MAX_TOKENS);
    }

    #[test]
    fn the_vision_envelope_can_be_lowered() {
        let a = args(&["--vision", "--vision-max-tokens", "8192"]);
        let config = expect_config(resolve(&a, no_env).expect("resolve"));
        assert_eq!(config.vision.map(|v| v.max_tokens()), Some(8192));
    }

    #[test]
    fn a_vision_envelope_without_vision_is_refused_rather_than_ignored() {
        let err = resolve(&args(&["--vision-max-tokens", "8192"]), no_env).expect_err("no vision");
        assert!(err.0.contains("--vision"), "{}", err.0);
        let env = env_map(&[("IGNIS_VISION_MAX_TOKENS", "8192")]);
        assert!(resolve(&[], env).is_err(), "the env form too");
    }

    #[test]
    fn a_vision_envelope_outside_the_range_is_refused_naming_it() {
        for raw in ["0", "1048577", "-1", "lots"] {
            let a = args(&["--vision", "--vision-max-tokens", raw]);
            let err = resolve(&a, no_env).expect_err("out of range");
            assert!(err.0.contains("--vision-max-tokens"), "{}", err.0);
            assert!(err.0.contains("1048576"), "{}", err.0);
            assert!(err.0.contains(raw), "{}", err.0);
        }
    }

    #[test]
    fn the_vision_env_vars_apply_and_the_flags_win_over_them() {
        let env = env_map(&[("IGNIS_VISION", "true"), ("IGNIS_VISION_MAX_TOKENS", "4096")]);
        let config = expect_config(resolve(&[], env).expect("resolve"));
        assert_eq!(config.vision.map(|v| v.max_tokens()), Some(4096));

        let env = env_map(&[("IGNIS_VISION", "true"), ("IGNIS_VISION_MAX_TOKENS", "4096")]);
        let a = args(&["--vision-max-tokens", "2048"]);
        let config = expect_config(resolve(&a, env).expect("resolve"));
        assert_eq!(config.vision.map(|v| v.max_tokens()), Some(2048), "flag must win over env");

        let env = env_map(&[("IGNIS_VISION", "false")]);
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).vision, None);

        let env = env_map(&[("IGNIS_VISION", "maybe")]);
        let err = resolve(&[], env).expect_err("bad bool");
        assert!(err.0.contains("IGNIS_VISION") && err.0.contains("maybe"), "{}", err.0);
    }

    #[test]
    fn help_lists_the_vision_flags() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(text.contains("--vision ") && text.contains("--vision-max-tokens"), "{text}");
    }

    #[test]
    fn vision_with_dflash2_is_refused_naming_both() {
        let a = args(&["--vision", "--spec", "dflash2", "--draft-tokens", "4"]);
        let err = resolve(&a, no_env).expect_err("vision + dflash2");
        assert!(err.0.contains("--vision") && err.0.contains("dflash2"), "{}", err.0);
        let env = env_map(&[("IGNIS_VISION", "true"), ("IGNIS_SPEC", "dflash2"), ("IGNIS_DRAFT_TOKENS", "4")]);
        assert!(resolve(&[], env).is_err(), "the env form too");
        // Each alone still loads.
        assert!(resolve(&args(&["--vision"]), no_env).is_ok());
        assert!(resolve(&args(&["--spec", "dflash2", "--draft-tokens", "4"]), no_env).is_ok());
    }

    // ── media acquisition (GitHub #179) ──────────────────────────────────

    #[test]
    fn media_defaults_to_no_private_network_and_a_one_gib_cache() {
        let config = expect_config(resolve(&args(&["--vision"]), no_env).expect("resolve"));
        assert_eq!(config.media, MediaOptions { allow_private_network: false, cache_bytes: 1024 << 20 });
        assert_eq!(expect_config(resolve(&[], no_env).expect("resolve")).media, MediaOptions::default());
    }

    #[test]
    fn media_flags_set_the_private_network_opt_in_and_the_cache() {
        let a = args(&["--vision", "--media-allow-private-network", "--media-cache-mib", "0"]);
        let config = expect_config(resolve(&a, no_env).expect("resolve"));
        assert_eq!(config.media, MediaOptions { allow_private_network: true, cache_bytes: 0 });

        let env = env_map(&[
            ("IGNIS_VISION", "true"),
            ("IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK", "true"),
            ("IGNIS_MEDIA_CACHE_MIB", "64"),
        ]);
        let config = expect_config(resolve(&args(&["--media-cache-mib", "128"]), env).expect("resolve"));
        assert_eq!(config.media, MediaOptions { allow_private_network: true, cache_bytes: 128 << 20 });
    }

    #[test]
    fn media_flags_without_vision_are_refused_rather_than_ignored() {
        for a in [&["--media-allow-private-network"][..], &["--media-cache-mib", "10"]] {
            let err = resolve(&args(a), no_env).expect_err("no vision");
            assert!(err.0.contains(a[0]) && err.0.contains("--vision"), "{}", err.0);
        }
    }

    #[test]
    fn a_media_cache_outside_the_range_is_refused_naming_it() {
        for raw in ["65537", "-1", "lots"] {
            let err = resolve(&args(&["--vision", "--media-cache-mib", raw]), no_env).expect_err("range");
            assert!(err.0.contains("--media-cache-mib") && err.0.contains(raw), "{}", err.0);
        }
        let env = env_map(&[("IGNIS_VISION", "true"), ("IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK", "maybe")]);
        assert!(resolve(&[], env).expect_err("bad bool").0.contains("maybe"));
    }

    #[test]
    fn help_lists_the_media_flags() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve") else {
            panic!("expected Help");
        };
        assert!(text.contains("--media-allow-private-network") && text.contains("--media-cache-mib"), "{text}");
    }

    #[test]
    fn help_lists_the_speculation_flags() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(text.contains("--spec") && text.contains("--draft-tokens"), "{text}");
    }

    // ── the Playground (GitHub #163, ADR 0026) ───────────────────────────

    #[test]
    fn the_playground_is_off_by_default_and_on_with_ui() {
        assert!(!expect_config(resolve(&[], no_env).expect("resolve")).ui);
        assert!(expect_config(resolve(&args(&["--ui"]), no_env).expect("resolve")).ui);
    }

    #[test]
    fn ui_takes_no_value_and_has_no_env_var() {
        // A bare switch: the next argument is parsed as a flag of its own.
        let config = expect_config(resolve(&args(&["--ui", "--bind", "b"]), no_env).expect("resolve"));
        assert!(config.ui);
        assert_eq!(config.bind, "b");

        let env = env_map(&[("IGNIS_UI", "true")]);
        assert!(!expect_config(resolve(&[], env).expect("resolve")).ui);
    }

    #[test]
    fn help_lists_the_ui_flag() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(text.contains("--ui"), "{text}");
    }

    // ── Prometheus metrics (GitHub #89, ADR 0017) ────────────────────────

    #[test]
    fn metrics_are_off_by_default_and_on_their_own_listener_with_metrics() {
        assert_eq!(expect_config(resolve(&[], no_env).expect("resolve")).metrics, None);
        assert_eq!(
            expect_config(resolve(&args(&["--metrics"]), no_env).expect("resolve")).metrics,
            Some(DEFAULT_METRICS_BIND.to_owned())
        );
        assert_ne!(DEFAULT_METRICS_BIND, DEFAULT_BIND);
    }

    #[test]
    fn metrics_bind_moves_the_metrics_listener_in_either_order() {
        for argv in [
            ["--metrics", "--metrics-bind", "127.0.0.1:9100"],
            ["--metrics-bind", "127.0.0.1:9100", "--metrics"],
        ] {
            let config = expect_config(resolve(&args(&argv), no_env).expect("resolve"));
            assert_eq!(config.metrics.as_deref(), Some("127.0.0.1:9100"), "{argv:?}");
        }
    }

    #[test]
    fn metrics_bind_without_metrics_or_value_or_on_the_api_address_is_refused() {
        let err = resolve(&args(&["--metrics-bind", "127.0.0.1:9100"]), no_env).unwrap_err();
        assert!(err.0.contains("--metrics"), "{err}");
        assert!(resolve(&args(&["--metrics", "--metrics-bind"]), no_env).is_err());
        let err = resolve(
            &args(&["--metrics", "--bind", "127.0.0.1:7000", "--metrics-bind", "127.0.0.1:7000"]),
            no_env,
        )
        .unwrap_err();
        assert!(err.0.contains("--bind"), "{err}");
    }

    #[test]
    fn metrics_is_a_bare_flag_with_no_env_var_and_no_alias() {
        // A bare switch: the next argument is parsed as a flag of its own.
        let config =
            expect_config(resolve(&args(&["--metrics", "--bind", "b"]), no_env).expect("resolve"));
        assert!(config.metrics.is_some());
        assert_eq!(config.bind, "b");

        for name in ["IGNIS_METRICS", "IGNIS_PROMETHEUS", "IGNIS_METRICS_BIND"] {
            let env = move |key: &str| (key == name).then(|| "127.0.0.1:9100".to_owned());
            assert_eq!(expect_config(resolve(&[], env).expect("resolve")).metrics, None, "{name}");
        }
        for alias in ["-M", "--prometheus", "--metrics=true"] {
            assert!(resolve(&args(&[alias]), no_env).is_err(), "`{alias}` is not a metrics alias");
        }
    }

    #[test]
    fn help_lists_the_metrics_flags() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(text.contains("--metrics ") && text.contains("--metrics-bind"), "{text}");
    }

    // ── the API key ──────────────────────────────────────────────────────

    #[test]
    fn the_api_key_is_unset_by_default_and_an_empty_value_stays_unset() {
        assert_eq!(expect_config(resolve(&[], no_env).expect("resolve")).api_key, None);
        let env = env_map(&[("IGNIS_API_KEY", "")]);
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).api_key, None);
    }

    #[test]
    fn the_api_key_resolves_flag_over_env() {
        let env = env_map(&[("IGNIS_API_KEY", "from-env")]);
        let config = expect_config(resolve(&[], &env).expect("resolve"));
        assert_eq!(config.api_key, Some(ApiKeySetting::Fixed(ApiKey::new("from-env"))));

        let config = expect_config(resolve(&args(&["--api-key", "from-flag"]), &env).expect("resolve"));
        assert_eq!(
            config.api_key,
            Some(ApiKeySetting::Fixed(ApiKey::new("from-flag"))),
            "flag must win over env"
        );
    }

    #[test]
    fn auto_asks_for_a_generated_key_from_the_flag_or_the_env() {
        let config = expect_config(resolve(&args(&["--api-key", "auto"]), no_env).expect("resolve"));
        assert_eq!(config.api_key, Some(ApiKeySetting::Generate));
        let env = env_map(&[("IGNIS_API_KEY", "auto")]);
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).api_key, Some(ApiKeySetting::Generate));
    }

    #[test]
    fn a_generated_key_is_fresh_and_long() {
        let a = ApiKey::generate().expect("random source");
        let b = ApiKey::generate().expect("random source");
        assert_ne!(a, b);
        let hex = a.as_str().strip_prefix("sk-ignis-").expect("prefix");
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()), "{hex}");
    }

    #[test]
    fn an_api_key_matches_only_itself_and_never_prints() {
        let key = ApiKey::new("sk-secret");
        assert!(key.matches("sk-secret"));
        for other in ["", "sk-secre", "sk-secret!", "sk-Secret"] {
            assert!(!key.matches(other), "{other:?}");
        }
        let config = expect_config(resolve(&args(&["--api-key", "sk-secret"]), no_env).expect("resolve"));
        assert!(!format!("{config:?}").contains("sk-secret"));
    }

    #[test]
    fn help_lists_the_api_key_flag() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(text.contains("--api-key") && text.contains("IGNIS_API_KEY"), "{text}");
    }

    // ── exposure (ADR 0028) ──────────────────────────────────────────────

    #[test]
    fn nothing_is_exposed_by_default() {
        let config = expect_config(resolve(&[], no_env).expect("resolve"));
        assert_eq!(config.expose, None);
        assert_eq!(config.api_key, None, "no exposure, no forced key");
    }

    #[test]
    fn expose_resolves_flag_over_env() {
        let config =
            expect_config(resolve(&args(&["--expose", "cloudflare-quick"]), no_env).expect("resolve"));
        assert_eq!(config.expose, Some(Expose::CloudflareQuick));

        let env = env_map(&[("IGNIS_EXPOSE", "cloudflare-quick")]);
        assert_eq!(expect_config(resolve(&[], &env).expect("resolve")).expose, Some(Expose::CloudflareQuick));
        let err = resolve(&args(&["--expose", "nope"]), &env).expect_err("the flag wins, and is checked");
        assert!(err.0.contains("nope"), "{err}");
    }

    #[test]
    fn an_unknown_expose_mode_is_a_usage_error() {
        let err = resolve(&args(&["--expose", "ngrok"]), no_env).expect_err("unknown mode");
        assert!(err.0.contains("--expose") && err.0.contains("ngrok"), "{err}");
        assert!(err.0.contains("cloudflare-quick"), "{err}");
    }

    #[test]
    fn exposing_without_a_key_generates_one() {
        let config =
            expect_config(resolve(&args(&["--expose", "cloudflare-quick"]), no_env).expect("resolve"));
        assert_eq!(config.api_key, Some(ApiKeySetting::Generate));
        // An empty key is no key.
        let env = env_map(&[("IGNIS_API_KEY", ""), ("IGNIS_EXPOSE", "cloudflare-quick")]);
        assert_eq!(expect_config(resolve(&[], env).expect("resolve")).api_key, Some(ApiKeySetting::Generate));
    }

    #[test]
    fn exposing_keeps_the_key_the_operator_chose() {
        let a = args(&["--expose", "cloudflare-quick", "--api-key", "sk-mine"]);
        let config = expect_config(resolve(&a, no_env).expect("resolve"));
        assert_eq!(config.api_key, Some(ApiKeySetting::Fixed(ApiKey::new("sk-mine"))));

        let env = env_map(&[("IGNIS_API_KEY", "sk-env")]);
        let config = expect_config(resolve(&args(&["--expose", "cloudflare-quick"]), env).expect("resolve"));
        assert_eq!(config.api_key, Some(ApiKeySetting::Fixed(ApiKey::new("sk-env"))));
    }

    #[test]
    fn help_lists_the_expose_flag() {
        let ConfigOutcome::Help(text) = resolve(&args(&["--help"]), no_env).expect("resolve")
        else {
            panic!("expected Help");
        };
        assert!(text.contains("--expose") && text.contains("cloudflare-quick"), "{text}");
    }
}
