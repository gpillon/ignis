//! `ignis-server`'s configuration: every field declared once (ADR 0046,
//! spec config-v2/01), resolved from the command line, the environment, a
//! config file and a profile (spec config-v2/02) in one precedence order,
//! per model family.
//!
//! The module is in layers, each in its own file:
//!
//! - `field.rs` / `kind.rs` — what a field is (its [`field::FieldMeta`]) and
//!   how a value of each kind is read and written;
//! - `schema.rs` — every field, declared once per group by a `macro_rules!`
//!   block, giving the grouped [`schema::Settings`];
//! - `source.rs` — the sources, gathered into layers, and the resolver that
//!   walks them in precedence order.
//!
//! This file turns resolved settings into the [`Config`] the rest of the
//! server reads — the composite values (`VramMode`, `Speculation`,
//! `Vision`, …) and the rules that tie one field to another (a vision
//! envelope needs vision on) — and fits a config to the model family an
//! artifact names.
//!
//! [`resolve`] is the seam: pure (no `std::env`, no filesystem, no process
//! exit), so precedence and validation are covered by fast unit tests
//! instead of only end-to-end runs.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::instruction::InstructionPolicy;
use crate::thinking::ReasoningEffort;

pub mod cli;
pub mod field;
pub mod file;
pub mod kind;
pub mod profile;
pub mod schema;
pub mod source;

#[cfg(test)]
mod tests;

use source::{FileSource, Fit, Resolution, Sources};

/// The default loaded-model id (the v1 specialization: Qwen 3.8-27B —
/// `CONTEXT.md`).
pub const DEFAULT_MODEL: &str = "qwen3.8-27b";

/// The default bind address: localhost, port 8000 (OpenAI convention).
pub const DEFAULT_BIND: &str = "127.0.0.1:8000";

/// Where `--server-metrics` serves Prometheus when `--server-metrics-bind`
/// names nowhere else (GitHub #89, ADR 0017): localhost, on the port
/// OpenTelemetry's Prometheus exporter uses — its own listener, never the
/// API's.
pub const DEFAULT_METRICS_BIND: &str = "127.0.0.1:9464";

/// The server-wide thinking budget a request that sets none runs under, in
/// reasoning tokens (spec server/08): 32,768, the owner's call of 2026-10-09.
/// It sits inside the 38,912 default cap less the 2,048-token answer reserve
/// (36,864), so a turn at the default cap is forced closed at 32,768 and
/// answers instead of reasoning past its cap; a smaller `max_tokens` clamps
/// it to that cap less the reserve. The 6,144 it replaced cost ~8-9 points
/// on GPQA Diamond (82.8% against ~91% unbudgeted on the first 22 questions).
/// `--model-thinking-budget off` turns it off.
pub const DEFAULT_THINKING_BUDGET: u32 = 32_768;

/// `--model-default-max-tokens`' default (ADR 0045, GitHub #309): what a
/// request that names no `max_tokens` may generate, its reasoning included,
/// on both models -- Qwen's recommended output length for complex tasks. `0`
/// is no default: such a request may generate to the end of the context.
pub const DEFAULT_MAX_TOKENS: u32 = 38_912;

/// The default non-streaming completion timeout, in seconds (GitHub #95).
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u32 = 30;

/// Whether the Playground is served without anyone saying so (GitHub #163,
/// ADR 0026). On: a route, and no cost to a server nobody opens in a browser.
pub const DEFAULT_UI: bool = true;

/// Whether a missing model may be fetched (GitHub #234, ADR 0033). On: a
/// server that cannot find its weights is useless, and the one machine that
/// must never spend the bandwidth says so with `--download-enabled false`.
pub const DEFAULT_MODEL_DOWNLOAD: bool = true;

/// Where a fetched model lands (`--download-path`): the same `./models`
/// every other instruction in this repo names, so a `hf download --local-dir
/// models` done by hand and a download the server did are the same file.
pub const DEFAULT_MODEL_DOWNLOAD_PATH: &str = "./models";

/// How long a model switch waits for the requests already running on the
/// old model before it cuts them, in seconds (`--switch-drain-timeout`,
/// spec model-switch/01): the grace window that gives a switch a finite
/// upper bound whatever the load.
pub const DEFAULT_SWITCH_DRAIN_TIMEOUT_SECS: u32 = 30;

/// The upper bound `--switch-drain-timeout` accepts: a ceiling against a
/// fat-fingered value, as [`MAX_REQUEST_TIMEOUT_SECS`] is for its flag.
pub const MAX_SWITCH_DRAIN_TIMEOUT_SECS: u32 = 3600;

/// Whether a request naming another model may switch the server to it
/// without anyone saying so (`--switch-allow-implicit`, spec model-switch/01
/// §Implicit switch). On: the owner's clients name the model they want and
/// expect it served, and only a model `--switch-known-models` lists can be
/// loaded this way, so the default moves the server to nothing the operator
/// did not name.
pub const DEFAULT_ALLOW_MODEL_SWITCH: bool = true;

/// The upper bound `--server-request-timeout` accepts: a ceiling against a
/// fat-fingered value, not a real operating point — a healthy request
/// legitimately runs for minutes at a large `max_tokens`, never hours.
pub const MAX_REQUEST_TIMEOUT_SECS: u32 = 3600;

/// The upper bound `--vision-embedding-pool-mib` accepts (GitHub #243): a
/// ceiling against a fat-fingered value, not the real limit. The real limit
/// is the VRAM plan — a pool the budget cannot hold fails the load by name
/// (ADR 0030) — and 64 GiB is past the largest card this engine runs on, so
/// this only catches a unit mistake.
pub const MAX_VISION_EMBEDDING_POOL_MIB: u64 = 64 * 1024;

// The prefill-chunk and per-sequence-context defaults live in
// `ignis_runtime` (re-exported below), the same numbers `CudaLeafConfig`
// falls back to — one source of truth for what `ignis-server` runs with
// when the operator passes no flags, rather than two constants that have
// to be kept in sync by hand across the crate boundary.
pub use ignis_runtime::{DEFAULT_MAX_CONTEXT, DEFAULT_PREFILL_CHUNK, PREFILL_CHUNK_ALIGNMENT};

pub use ignis_core::{KvFormat, KvPoolSize, VramMode};
use ignis_core::compute::ModelFamily;

/// `--vram-headroom-bytes`' default: what a derived VRAM budget leaves to the
/// desktop and every other process on the card (GitHub #210). 1.5 GiB since
/// 2026-10-09: at 1 GiB a browser drawing a page made WDDM page the model out
/// (the 27B 19 -> 70 ms a decode round, Flash-Next 4.6 tok/s), at 1.5 GiB
/// neither moved (finding 2026-10-09-vram-headroom-wddm-paging).
pub const DEFAULT_VRAM_HEADROOM_BYTES: u64 = 1536 * 1024 * 1024;
pub use ignis_core::{MAX_DRAFT_TOKENS, ProposalHead, Speculation, SpeculativeBackend};
use ignis_core::speculation::FLASH_NEXT_DEFAULT_DRAFT_TOKENS;

pub use ignis_core::{MAX_YARN_FACTOR, RopeScaling};
pub use ignis_core::{DEFAULT_VISION_MAX_TOKENS, VISION_MAX_TOKENS_LIMIT, Vision};

pub use crate::expose::Expose;

/// `--media-cache-mib`'s default (the reference's 1 GiB).
pub const DEFAULT_MEDIA_CACHE_MIB: u32 = 1024;

/// The largest `--media-cache-mib` accepted: a ceiling against a
/// fat-fingered value (64 GiB of host memory for prepared patches).
pub const MEDIA_CACHE_MIB_LIMIT: u32 = 64 * 1024;

/// `--reuse-kv-host-pool-bytes`' default (P4-07, GitHub #125): comfortably
/// holds several full-context snapshots (each ~528 MB per ADR 0024's
/// estimate) without an operator having to reason about the format's
/// per-snapshot cost just to start the server.
///
/// Since GitHub #213 (ADR 0030) it is page-locked whole at start rather than
/// blob by blob while serving, so it is RAM the process holds even idle.
pub const DEFAULT_HOST_POOL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// `--reuse-prompt`'s default (GitHub #186, ADR 0029): on. Cross-request
/// reuse is the owner's workload — an agent's tool loop re-sends its whole
/// history every iteration — so it is what the engine does unless asked not
/// to.
pub const DEFAULT_PROMPT_REUSE: bool = true;

/// `--reuse-retained-device`'s and `--reuse-retained-host`'s defaults with
/// prompt reuse on (GitHub #281): no slot in VRAM, two per decode lane in the
/// pinned host block. With prompt reuse off both default to 0.
pub use ignis_runtime::{DEFAULT_RETAINED_DEVICE_SLOTS, DEFAULT_RETAINED_HOST_SLOTS};

/// `--reuse-retained-interactive-ttl`'s default, in seconds (GitHub #190):
/// the scheduler's own starting value, restated as a flag default rather
/// than chosen twice.
pub const DEFAULT_RETAINED_INTERACTIVE_TTL_SECS: u32 =
    ignis_core::host::DEFAULT_RETAINED_INTERACTIVE_TTL.as_secs() as u32;

/// The fully-resolved config `main` needs to start the server: the
/// declared fields of [`schema::Settings`], with the ones that only mean
/// something together already combined (`vram`, `speculation`, `vision`,
/// `media`, `metrics`, `instruction_policy`, `ngram_cache`) and checked
/// against each other.
///
/// Built only by resolution ([`resolve`], [`Config::for_family`],
/// [`fit_to_family`]), never field by field, so its combined values always
/// agree with the [`Config::basis`] they were derived from. A copy made with
/// struct-update syntax (`Config { kv_format, ..base }`, as tests do to vary
/// one knob) keeps its base's basis: it is a fine input to anything that
/// reads its fields, and re-resolving it starts again from the base's
/// sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// The id the model is served under: the operator's (`--model-id`), the
    /// loaded model's own once a family is known, or [`DEFAULT_MODEL`]
    /// before.
    pub model: String,
    /// Whether the operator named [`Config::model`] (`--model-id`).
    /// Unnamed, a load is served under its own model's id
    /// ([`served_model_for`]), not the 27B's default.
    pub model_named: bool,
    pub bind: String,
    pub artifact: Option<PathBuf>,
    /// May the server fetch [`Config::model`] when no artifact is on disk
    /// (`--download-enabled`, GitHub #234)? On by default. Off never refuses
    /// a start: it falls back to the placeholder template, exactly as an
    /// unfetchable model always did. Only consulted when `artifact` is
    /// `None` — a named path is the operator's word.
    pub model_download: bool,
    /// Where a fetched model lands, and where one fetched earlier is looked
    /// for (`--download-path`, GitHub #234). Flat, one file per model: the
    /// artifact and its sidecar keep the names the repo publishes them under.
    pub model_download_path: PathBuf,
    /// Flash-Next's n-gram hot-row cache between loads (`--ngram-persist` /
    /// `--ngram-persist-path`): on, beside the model, unless the operator
    /// says otherwise.
    pub ngram_cache: ignis_core::ngram_cache::PersistenceOptions,
    /// Flash-Next's n-gram hot-row budget (`--ngram-hot-bytes`, GitHub
    /// #306): a size, or `auto` for what the host plan leaves. `None`: the
    /// 1 GiB default. The 27B has no n-gram table and refuses it.
    pub ngram_hot_bytes: Option<ignis_core::ngram_table::HotBudget>,
    /// KV-disk, Tier 2 (spec vram-budget/03): its budget in bytes
    /// (`--kv-disk-bytes`), a ceiling cut at start to the volume's free space
    /// above its 10 GiB margin. `None`: the model family's (4 GiB on
    /// Flash-Next, 0 = off on the 27B).
    pub kv_disk_bytes: Option<u64>,
    /// Where KV-disk's files go (`--kv-disk-path`): the n-gram cache's rule
    /// -- beside the model (the default), `auto`'s per-user cache directory
    /// under `kv-disk`, or a named directory.
    pub kv_disk_location: ignis_core::ngram_cache::CacheLocation,
    pub enable_thinking: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    /// The server-wide thinking budget (`--model-thinking-budget`, default
    /// [`DEFAULT_THINKING_BUDGET`]): the reasoning tokens a request may spend
    /// before the model's close is forced. `None` = no budget (`off`).
    pub thinking_budget: Option<u32>,
    /// The prefill chunk width, in tokens (a nonzero multiple of
    /// [`PREFILL_CHUNK_ALIGNMENT`]).
    pub prefill_chunk: u32,
    /// The **decode share** (`--model-decode-share`, GitHub #306): the
    /// percent of the model's time decoding lanes keep while a prompt
    /// prefills, 0-99. `None`: the model family's own.
    pub decode_share_percent: Option<u32>,
    /// The maximum per-sequence context, in tokens (the largest prompt +
    /// generation budget a single request may reserve).
    pub max_context: u32,
    /// The **default `max_tokens`** (`--model-default-max-tokens`, ADR
    /// 0045): the generation cap of a request that sends none, clamped by
    /// the scheduler to what its prompt leaves of the context.
    /// [`DEFAULT_MAX_TOKENS`] unless named; `0` is none.
    pub default_max_tokens: u32,
    /// The KV storage format this load runs on (ADR 0022, GitHub #122),
    /// fixed for the life of the load.
    pub kv_format: KvFormat,
    /// The KV pool the operator named (`--vram-kv-pool-bytes`, ADR 0045): a
    /// byte count, or a token count (`512Ktok`). Only parsed here: the load's
    /// plan, which knows the model's bytes per token, turns it into pages and
    /// refuses one smaller than a full context and a page per retained slot.
    /// `None` (the default) takes the KV pool policy's size.
    pub kv_pool: Option<KvPoolSize>,
    /// How the load's **VRAM budget** is chosen (GitHub #210, ADR 0030):
    /// derived from `--vram-headroom-bytes` (the default, with
    /// [`DEFAULT_VRAM_HEADROOM_BYTES`]) or named by `--vram-budget-bytes`,
    /// with `--vram-allow-oversubscription` only beside the latter.
    pub vram: VramMode,
    /// Start Flash-Next with an expert cache below its 12 GiB floor, with a
    /// warning (`--vram-allow-expert-cache-below-floor`, ADR 0045). The 27B,
    /// which has no expert cache, refuses it.
    pub allow_expert_cache_below_floor: bool,
    /// The KV-RAM host tier's budget, in bytes (P4-07, GitHub #125): pinned
    /// host memory for evicted (suspended) request snapshots. `0` disables
    /// the tier (admission refuses instead of evicting once the resident
    /// lanes are full). A byte budget, not a page or lane count, because a
    /// snapshot's fixed GDN floor (~145 MiB) is paid regardless of prompt
    /// length.
    pub host_pool_bytes: u64,
    /// Cross-request state reuse (`--reuse-prompt`, GitHub #186, ADR 0029).
    /// On by default. Off means a request captures no prompt checkpoint and
    /// claims none, so a cold bench measures a cold engine and a correctness
    /// oracle prefills every prompt it is given.
    pub prompt_reuse: bool,
    /// The load's retained slots (GitHub #215, #281, ADR 0030): places for
    /// one mutable-state image each, reserved at load, where every prompt
    /// checkpoint and shared prefix keeps its image. Device slots
    /// (`--reuse-retained-device`, [`DEFAULT_RETAINED_DEVICE_SLOTS`]) sit in
    /// the device state arenas and are handed out first; host slots
    /// (`--reuse-retained-host`, [`DEFAULT_RETAINED_HOST_SLOTS`]) sit in one
    /// pinned host block and cost a PCIe copy per capture and per claim. Both
    /// 0 with prompt reuse off unless named.
    pub retained_device_slots: u32,
    pub retained_host_slots: u32,
    /// Whether [`Config::retained_host_slots`] was given at all — by the
    /// operator or by a profile. Ungiven, a Flash-Next load takes its own
    /// default (spec flash-next/05), not the 27B's.
    pub retained_host_named: bool,
    /// How long a retained Interactive checkpoint in KV-RAM keeps its class's
    /// priority after its conversation last used it, in seconds
    /// (`--reuse-retained-interactive-ttl`, GitHub #190). Past it the entry
    /// ranks as an Agent's would.
    pub retained_interactive_ttl_secs: u32,
    /// Where `system` and `developer` messages go before the conversation is
    /// templated (`--server-system-message-policy` /
    /// `--server-developer-message-policy`, GitHub #209).
    pub instruction_policy: InstructionPolicy,
    /// Speculative decoding, chosen at load (`--spec-backend` /
    /// `--spec-draft-tokens`, P5-02 GitHub #150). `None` loads nothing of the
    /// drafter; Flash-Next's MTP head too is off unless `--spec-backend mtp`
    /// names it (spec flash-next/07).
    pub speculation: Option<Speculation>,
    /// `--spec-backend off`: no speculation, the default one included
    /// (GitHub #307).
    pub speculation_off: bool,
    /// Flash-Next's draft row budget (`--spec-draft-rows`, GitHub #307): a
    /// round of w lanes verifies min(draft tokens, rows / w - 1) drafts per
    /// lane. `None`: the decode route's 8 rows.
    pub draft_rows: Option<u32>,
    /// Flash-Next's decode lanes (`--spec-decode-lanes`, GitHub #306),
    /// 1..=[`ignis_core::N_DECODE_LANES`]. `None`: the engine's default of 3.
    /// The 27B has a fixed lane count and refuses the flag.
    pub decode_lanes: Option<u32>,
    /// Vision, chosen at load (`--vision-enabled` / `--vision-max-tokens`,
    /// GitHub #177). `None` binds and reserves nothing of the vision tower.
    pub vision: Option<Vision>,
    /// The text rotary table, chosen at load (`--model-rope-scaling`, GitHub
    /// #227). [`RopeScaling::NONE`] is the linear table the engine has always
    /// used; a YaRN factor rescales the checkpoint's trained 262,144-position
    /// envelope, which is what a context past it needs.
    pub rope_scaling: RopeScaling,
    /// Media acquisition (`--media-allow-private-network`,
    /// `--media-cache-mib`, GitHub #179). Only nameable with vision on.
    pub media: MediaOptions,
    /// How long a non-streaming request waits for its completion before the
    /// handler gives up with a `504` (GitHub #95). In `[1, MAX_REQUEST_TIMEOUT_SECS]`.
    pub request_timeout_secs: u32,
    /// How long a model switch (spec model-switch/01) waits for the old
    /// model's in-flight requests before it cancels them
    /// (`--switch-drain-timeout`). In `[0, MAX_SWITCH_DRAIN_TIMEOUT_SECS]`;
    /// 0 cuts them at once.
    pub switch_drain_timeout_secs: u32,
    /// Whether a request whose `model` names another model listed in
    /// [`Config::known_models`] switches the server to it, and is then
    /// served on it (`--switch-allow-implicit`, spec model-switch/01
    /// §Implicit switch; [`DEFAULT_ALLOW_MODEL_SWITCH`]).
    pub allow_model_switch: bool,
    /// The models a request may switch to by naming them, each with the
    /// artifact it loads from (`--switch-known-models <id>=<path>`,
    /// repeatable). The operator's entries only: the model the server starts
    /// on joins them once its load has said which id it serves under.
    pub known_models: BTreeMap<String, PathBuf>,
    /// Serve the Playground under `/ui/` (GitHub #163, ADR 0026). On unless
    /// `--server-ui false` turns it off: a binary that embedded the build
    /// serves it, and one that did not serves the page saying how to build
    /// it, so the default costs a route and nothing else.
    pub ui: bool,
    /// The metrics listener's address when `--server-metrics` is on (GitHub
    /// #89, ADR 0017): `--server-metrics-bind`, else
    /// [`DEFAULT_METRICS_BIND`]. `None` = metrics off.
    pub metrics: Option<String>,
    /// The key every `/v1` request must present as `Authorization: Bearer
    /// <key>` (`--server-api-key`). `None` (the default) keeps the API open,
    /// as it has always been on localhost.
    pub api_key: Option<ApiKeySetting>,
    /// How the server is exposed beyond its bind address (`--server-expose`,
    /// ADR 0028). `Some` always comes with an API key: without one,
    /// resolution sets `api_key` to [`ApiKeySetting::Generate`].
    pub expose: Option<Expose>,
    /// What this config was resolved from: the gathered sources, the
    /// settings resolved from them, and the family they were resolved for.
    /// [`Config::for_family`], [`fit_to_family`] and [`Config::with_patch`]
    /// start again from here; `GET /v1/config` shows its settings.
    pub basis: Basis,
}

/// What a [`Config`] was resolved from (see [`Config::basis`]). Shared, so a
/// config clones in O(1) whatever its sources hold.
#[derive(Clone, PartialEq, Eq)]
pub struct Basis(Arc<BasisInner>);

#[derive(PartialEq, Eq)]
struct BasisInner {
    sources: Sources,
    resolution: Resolution,
    family: Option<ModelFamily>,
}

impl Basis {
    /// The sources gathered at start, with any live patch on top.
    pub fn sources(&self) -> &Sources {
        &self.0.sources
    }

    /// Every declared field's resolved value — spec config-v2/01's
    /// "resolved Config".
    pub fn settings(&self) -> &schema::Settings {
        &self.0.resolution.settings
    }

    /// Where each field's value came from.
    pub fn resolution(&self) -> &Resolution {
        &self.0.resolution
    }

    /// The family the settings were resolved for; `None` before an artifact
    /// named one.
    pub fn family(&self) -> Option<ModelFamily> {
        self.0.family
    }
}

impl std::fmt::Debug for Basis {
    /// The family and which spellings each source set, never a value (the
    /// sources hold the API key's raw text).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Basis").field("family", &self.0.family).field("sources", &self.0.sources).finish()
    }
}

/// The speculative backend a config names (`--spec-backend`): one, or `off`
/// — which is not the same as naming none, since `off` also turns off a
/// backend a family would otherwise start by default (GitHub #307).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecChoice {
    Off,
    Backend(SpeculativeBackend),
}

/// How image parts are acquired (GitHub #179).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaOptions {
    /// Fetch image URLs that resolve to private, loopback, link-local,
    /// multicast or CGNAT addresses. Off by default, so an exposed server
    /// cannot be used to probe the operator's LAN.
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

/// What `--server-api-key` asked for: a key the operator chose, or `auto` —
/// one `main` generates at start and prints, the only time a key is printed.
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

/// What [`resolve`] produced: a runnable config, or what to do instead of
/// serving — print text, or write a file (`config generate` / `patch`) —
/// before exiting. `resolve` never prints, writes or exits itself; that
/// stays in `main`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigOutcome {
    Config(Config),
    Help(String),
    Version(String),
    /// Text for stdout (`help --fields`, `config print`, `config generate`
    /// to stdout or `--dry-run`).
    Print(String),
    /// A config file to write (`config generate --out`, `config patch`).
    Write { path: PathBuf, contents: String },
}

/// A configuration the server refuses: an unrecognized flag, a value that
/// does not parse or validate, two fields that contradict each other. The
/// message names the spelling the bad value arrived under and is suitable
/// for `eprintln!("ignis-server: {err}")` before exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Resolve `args` (argv without the program name) and `env` (injected so
/// tests never touch the real process environment) into a [`ConfigOutcome`],
/// with no filesystem: a config file is neither found nor read
/// ([`file::NoFiles`]). [`resolve_with`] is the same with one.
pub fn resolve(args: &[String], env: impl Fn(&str) -> Option<String>) -> Result<ConfigOutcome, ConfigError> {
    resolve_with(args, env, &file::NoFiles)
}

/// The env var naming the config file, as `--config` does.
pub const CONFIG_ENV: &str = "IGNIS_CONFIG";

/// The env var naming the hardware profile, as `--profile` does.
pub const PROFILE_ENV: &str = "IGNIS_PROFILE";

/// Resolve the command line against `env` and `files`.
///
/// A first word that is not a flag is a verb (`help`, `version`, `config
/// generate|print|patch`, [`cli`]); otherwise the server starts — the bare
/// invocation is unchanged, so `make start`'s command line needs no leading
/// word. `--help`/`-h` and `--version`/`-V` short-circuit before any other
/// flag is parsed or validated — `ignis-server --help --nonsense` just
/// prints help.
///
/// Every family-scoped value is resolved here too, for each family, so a
/// mistake in one (`--qwen38flashnext-model-max-context lots`) is refused at
/// start rather than at the first load of that family. The family checks
/// that depend on which artifact loads (the 27B's attention envelope) wait
/// for [`Config::for_family`].
pub fn resolve_with(
    args: &[String],
    env: impl Fn(&str) -> Option<String>,
    files: &dyn file::Files,
) -> Result<ConfigOutcome, ConfigError> {
    match args.first().map(String::as_str) {
        Some("help") => return cli::help(&args[1..]),
        Some("version") => return Ok(ConfigOutcome::Version(version_text())),
        Some("config") => return cli::config(&args[1..], &env, files),
        Some(word) if !word.starts_with('-') => {
            return Err(ConfigError(format!(
                "unknown command `{word}` (the commands are help, version and config; with none, the server starts)"
            )));
        }
        _ => {}
    }
    for arg in args {
        match arg.as_str() {
            "--help" | "-h" => return Ok(ConfigOutcome::Help(help_text())),
            "--version" | "-V" => return Ok(ConfigOutcome::Version(version_text())),
            _ => {}
        }
    }
    let parsed = source::parse_args(args)?;
    let named = parsed.config.clone().or_else(|| env(CONFIG_ENV).filter(|path| !path.is_empty()));
    let choice = match named {
        Some(path) => FileChoice::Explicit(PathBuf::from(path)),
        None => FileChoice::Discover,
    };
    let (sources, _) = gather(parsed, &env, files, choice)?;
    Config::from_sources(sources).map(ConfigOutcome::Config)
}

/// Which config file a resolution reads.
#[derive(Debug, Clone)]
pub(crate) enum FileChoice {
    /// None.
    None,
    /// The one `--config` / `IGNIS_CONFIG` named: it must exist — an
    /// operator who named a file meant it to.
    Explicit(PathBuf),
    /// The first of the conventional places that holds one, if any (spec
    /// config-v2/02 §Config-file auto-discovery). Only when nothing named a
    /// file: a named one short-circuits discovery entirely, never even
    /// looking at the candidates.
    Discover,
    /// This one if it exists (`config print`).
    IfPresent(PathBuf),
    /// This one, which must exist (`config patch`, which changes a file and
    /// does not make one).
    Required(PathBuf),
}

/// Gather every source of a resolution: the flags parsed, the environment,
/// the config file `choice` picks, and the hardware profile (spec
/// config-v2/02 §`--profile`) — named by `--profile`, `IGNIS_PROFILE` or the
/// file's own `profile:`, in that order, else [`profile::DEFAULT_PROFILE`];
/// looked up among the file's `profiles:` first and the built-in ones after,
/// by one function. Returns the file's document as written too, for a
/// caller that merges a change into it.
pub(crate) fn gather(
    parsed: source::ParsedArgs,
    env: &dyn Fn(&str) -> Option<String>,
    files: &dyn file::Files,
    choice: FileChoice,
) -> Result<(Sources, Option<serde_json::Value>), ConfigError> {
    let (file_source, loaded) = match choice {
        FileChoice::None => (FileSource::None, None),
        FileChoice::Explicit(path) => {
            let loaded = file::load(files, &path)?;
            (FileSource::Explicit(path), Some(loaded))
        }
        FileChoice::Discover => match file::discovery_candidates(env).into_iter().find(|path| files.exists(path)) {
            Some(path) => {
                let loaded = file::load(files, &path)?;
                (FileSource::Discovered(path), Some(loaded))
            }
            None => (FileSource::None, None),
        },
        FileChoice::IfPresent(path) if files.exists(&path) => {
            let loaded = file::load(files, &path)?;
            (FileSource::Explicit(path), Some(loaded))
        }
        FileChoice::IfPresent(_) => (FileSource::None, None),
        FileChoice::Required(path) => {
            if !files.exists(&path) {
                return Err(ConfigError(format!(
                    "no config file at {}: `config patch` changes a file; `config generate --out {}` makes one",
                    path.display(),
                    path.display()
                )));
            }
            let loaded = file::load(files, &path)?;
            (FileSource::Explicit(path), Some(loaded))
        }
    };
    let (value, document) = match loaded {
        Some((value, document)) => (Some(value), document),
        None => (None, file::Document::default()),
    };
    let profile_name = parsed
        .profile
        .or_else(|| env(PROFILE_ENV).filter(|name| !name.is_empty()))
        .or(document.profile)
        .unwrap_or_else(|| profile::DEFAULT_PROFILE.to_owned());
    let profile = profile::lookup(&profile_name, &document.profiles)?;
    let sources = Sources {
        patch: source::Layer::default(),
        flags: parsed.flags,
        env: source::env_layer(env)?,
        file: document.values,
        file_source,
        profile,
        profile_name,
    };
    Ok((sources, value))
}

impl Config {
    /// The config `sources` resolve to before any family is known, every
    /// family's scoped values checked on the way.
    pub fn from_sources(sources: Sources) -> Result<Config, ConfigError> {
        for family in field::FAMILIES {
            derive(&sources, Some(family), Fit::Switch, false)?;
        }
        derive(&sources, None, Fit::Start, false).map(|(config, _)| config)
    }

    /// The settings every field resolved to (`GET /v1/config`'s source).
    pub fn settings(&self) -> &schema::Settings {
        self.basis.settings()
    }

    /// This config's sources resolved again for an artifact of `family`, at
    /// **start** (spec flash-next/04): the family-scoped values apply, and an
    /// explicit value for a field `family` cannot take is refused by name —
    /// the operator named it for this load. Also refused: a served id naming
    /// the other model, a speculative backend the family does not draft
    /// with, and on the 27B a context past its attention's envelope.
    ///
    /// The returned config's [`Config::model`] is the id the load is served
    /// under.
    pub fn for_family(&self, family: ModelFamily) -> Result<Config, ConfigError> {
        derive(self.basis.sources(), Some(family), Fit::Start, true).map(|(config, _)| config)
    }

    /// This config with `patch`'s values over everything else — a live
    /// `PATCH /v1/config`, or a switch target's own artifact and id — and
    /// resolved again for the same family. Validates exactly as a start
    /// would: the patch is refused whole if any of it is wrong.
    pub fn with_patch(&self, patch: &source::Layer) -> Result<Config, ConfigError> {
        let mut sources = self.basis.sources().clone();
        sources.patch.overlay(patch);
        let config = Config::from_sources(sources)?;
        match self.basis.family() {
            Some(family) => config.for_family(family),
            None => Ok(config),
        }
    }
}

/// The patch naming a model switch's target (spec model-switch/01): its
/// artifact and the id it is served under, over everything else — the form
/// [`Config::with_patch`] takes, so the target's options are resolved again
/// from their sources like any other change rather than set on a struct.
pub fn target_patch(artifact: &std::path::Path, model: &str) -> Result<source::Layer, ConfigError> {
    let mut patch = source::Layer::default();
    let field = |name| schema::field("model", name).expect("a declared field");
    patch.set(field("artifact"), None, source::Candidate { raw: artifact.display().to_string(), spelling: "the switch target's artifact".to_owned() })?;
    patch.set(field("id"), None, source::Candidate { raw: model.to_owned(), spelling: "the switch target's model".to_owned() })?;
    Ok(patch)
}

/// The id a load of `family` is served under, or why `config` cannot start
/// on it ([`Config::for_family`]).
pub fn served_model_for(config: &Config, family: ModelFamily) -> Result<String, ConfigError> {
    config.for_family(family).map(|fitted| fitted.model)
}

/// `config` fitted to an artifact of `family` for a **model switch** (spec
/// model-switch/01), and the flags it dropped to get there.
///
/// One process serves the 27B and Flash-Next in turn on the options it was
/// started with, and some of those name a capability only one of the two
/// has: vision (the 27B's tower), a speculative backend (each model drafts
/// with its own), the Flash-Next-only knobs. At start such a value on the
/// wrong model is refused ([`Config::for_family`]) because the operator
/// named it for that load; on a switch it was named for the *other* model,
/// and refusing it would make the switch impossible on the options the owner
/// starts with. So it is dropped for this load — and returned, so the switch
/// says which. Which fields those are is each field's own
/// [`field::Applicability`], not a list kept here. A value both models take
/// but bound differently (`--model-max-context` past the 27B's attention
/// envelope) is not dropped: it is refused, as at start.
///
/// The family's scoped values apply here as at start: a switch is how the
/// 27B and Flash-Next each keep their own resource shape (spec
/// config-v2/01).
pub fn fit_to_family(config: &Config, family: ModelFamily) -> Result<(Config, Vec<String>), ConfigError> {
    derive(config.basis.sources(), Some(family), Fit::Switch, true)
}

/// Resolve `sources` for `family` and combine the result into a [`Config`]:
/// the cross-field rules first, then — `family_checks` — what the artifact's
/// family decides (the served id, the drafter, the attention envelope).
fn derive(
    sources: &Sources,
    family: Option<ModelFamily>,
    fit: Fit,
    family_checks: bool,
) -> Result<(Config, Vec<String>), ConfigError> {
    let resolution = source::resolve_settings(sources, family, fit)?;
    let mut dropped = resolution.dropped.clone();
    let s = &resolution.settings;
    // The spelling a field's value came under, for an error that names it;
    // its flag when it is at its default.
    let spelled = |group: &str, name: &str| {
        resolution
            .origin(group, name)
            .map(|origin| origin.spelling.clone())
            .unwrap_or_else(|| schema::field(group, name).expect("a declared field").flag())
    };
    let explicit = |group: &str, name: &str| resolution.explicit(group, name);

    let vram = derive_vram(&resolution, &spelled)?;

    let prompt_reuse = s.reuse.prompt;
    if explicit("reuse", "retained_interactive_ttl") && !prompt_reuse {
        return Err(ConfigError(format!(
            "`{} {}` requires `--reuse-prompt on` (nothing is retained without it)",
            spelled("reuse", "retained_interactive_ttl"),
            s.reuse.retained_interactive_ttl
        )));
    }

    let (speculation, speculation_off) = derive_speculation(&resolution, &spelled)?;
    let vision = derive_vision(&resolution, &spelled)?;
    let media = derive_media(&resolution, vision.is_some(), &spelled)?;

    // `--server-metrics` (GitHub #89, ADR 0017) opens its own listener;
    // naming its address without turning metrics on is refused rather than
    // ignored, and it can never share the API's.
    if explicit("server", "metrics_bind") && !s.server.metrics {
        return Err(ConfigError(format!(
            "`{}` requires `--server-metrics` (metrics are off without it)",
            spelled("server", "metrics_bind")
        )));
    }
    let metrics = s.server.metrics.then(|| s.server.metrics_bind.clone());
    if metrics.as_deref() == Some(s.server.bind.as_str()) {
        return Err(ConfigError(format!(
            "`{} {}` is the API's `--server-bind`: metrics need their own listener",
            spelled("server", "metrics_bind"),
            s.server.bind
        )));
    }
    // An exposed API is never open: the operator's key if they named one,
    // otherwise the same generated key `--server-api-key auto` gives.
    let api_key = match (&s.server.expose, &s.server.api_key) {
        (Some(_), None) => Some(ApiKeySetting::Generate),
        (_, api_key) => api_key.clone(),
    };

    let mut model = s.model.id.clone().unwrap_or_else(|| DEFAULT_MODEL.to_owned());
    let mut speculation = speculation;
    if let (true, Some(family)) = (family_checks, family) {
        let fitted = FamilyFit { model: &mut model, speculation: &mut speculation, dropped: &mut dropped };
        fit_family(fitted, &resolution, family, fit, &spelled)?;
    }

    let config = Config {
        model,
        model_named: s.model.id.is_some(),
        bind: s.server.bind.clone(),
        artifact: s.model.artifact.clone(),
        model_download: s.download.enabled,
        model_download_path: s.download.path.clone(),
        ngram_cache: ignis_core::ngram_cache::PersistenceOptions {
            enabled: s.ngram.persist,
            location: s.ngram.persist_path.clone(),
        },
        ngram_hot_bytes: s.ngram.hot_bytes,
        kv_disk_bytes: s.kv_disk.bytes,
        kv_disk_location: s.kv_disk.path.clone(),
        enable_thinking: s.model.enable_thinking,
        reasoning_effort: s.model.reasoning_effort,
        thinking_budget: s.model.thinking_budget,
        prefill_chunk: s.model.prefill_chunk,
        decode_share_percent: s.model.decode_share,
        max_context: s.model.max_context,
        default_max_tokens: s.model.default_max_tokens,
        kv_format: s.model.kv_format,
        kv_pool: s.vram.kv_pool_bytes,
        vram,
        allow_expert_cache_below_floor: s.vram.allow_expert_cache_below_floor,
        host_pool_bytes: s.reuse.kv_host_pool_bytes,
        prompt_reuse,
        retained_device_slots: s
            .reuse
            .retained_device
            .unwrap_or(if prompt_reuse { DEFAULT_RETAINED_DEVICE_SLOTS } else { 0 }),
        retained_host_slots: s
            .reuse
            .retained_host
            .unwrap_or(if prompt_reuse { DEFAULT_RETAINED_HOST_SLOTS } else { 0 }),
        retained_host_named: s.reuse.retained_host.is_some(),
        retained_interactive_ttl_secs: s.reuse.retained_interactive_ttl,
        instruction_policy: InstructionPolicy {
            system: s.server.system_message_policy,
            developer: s.server.developer_message_policy,
        },
        speculation,
        speculation_off,
        draft_rows: s.spec.draft_rows,
        decode_lanes: s.spec.decode_lanes,
        vision,
        rope_scaling: s.model.rope_scaling,
        media,
        request_timeout_secs: s.server.request_timeout,
        switch_drain_timeout_secs: s.switch.drain_timeout,
        allow_model_switch: s.switch.allow_implicit,
        known_models: s.switch.known_models.clone(),
        ui: s.server.ui,
        metrics,
        api_key,
        expose: s.server.expose,
        basis: Basis(Arc::new(BasisInner { sources: sources.clone(), resolution: resolution.clone(), family })),
    };
    Ok((config, dropped))
}

/// Say where the configuration came from (spec config-v2/02 AC 12):
/// `ignis.config.source`, naming `explicit` (the path `--config` /
/// `IGNIS_CONFIG` gave), `discovered` (the conventional path it was found
/// at) or `none`, with the profile in use. `main` logs it before the model
/// load begins, so it is there even when the load then fails — "why is this
/// value what it is" is answered by the first lines of the log.
pub fn log_source(config: &Config) {
    let sources = config.basis.sources();
    let path = sources.file_source.path().map(|path| path.display().to_string());
    tracing::info!(
        name: "ignis.config.source",
        source = sources.file_source.kind(),
        path = path.as_deref().unwrap_or("none"),
        profile = %sources.profile_name,
        "configuration source"
    );
}

/// The values [`fit_family`] may change: the served id, the speculation a
/// switch drops, and the list of what it dropped.
struct FamilyFit<'a> {
    model: &'a mut String,
    speculation: &'a mut Option<Speculation>,
    dropped: &'a mut Vec<String>,
}

/// The VRAM budget's mode (GitHub #210, ADR 0030): `--vram-headroom-bytes`
/// derives it, `--vram-budget-bytes` names it, `--vram-allow-oversubscription`
/// accepts a named one above free memory.
///
/// A headroom and a budget are two answers to one question, so both named
/// by the operator — from any mix of flags, env and file — is refused rather
/// than one silently winning. A profile's value is a default: an operator's
/// headroom wins over a profile's budget, and an operator's budget over a
/// profile's headroom. Oversubscription without a budget is refused too: a
/// derived budget is below free memory by construction.
fn derive_vram(resolution: &Resolution, spelled: &impl Fn(&str, &str) -> String) -> Result<VramMode, ConfigError> {
    let s = &resolution.settings.vram;
    let headroom_named = resolution.explicit("vram", "headroom_bytes");
    let budget_named = resolution.explicit("vram", "budget_bytes");
    if headroom_named && budget_named {
        return Err(ConfigError(format!(
            "`{}` and `{}` are mutually exclusive: a headroom derives the VRAM budget, a budget names it",
            spelled("vram", "headroom_bytes"),
            spelled("vram", "budget_bytes")
        )));
    }
    let budget = s.budget_bytes.filter(|_| !headroom_named);
    match budget {
        None => {
            if s.allow_oversubscription {
                return Err(ConfigError(format!(
                    "`{}` requires `--vram-budget-bytes` (a budget derived from free memory never exceeds it)",
                    spelled("vram", "allow_oversubscription")
                )));
            }
            Ok(VramMode::Derived { headroom_bytes: s.headroom_bytes })
        }
        Some(budget_bytes) => Ok(VramMode::Explicit { budget_bytes, allow_oversubscription: s.allow_oversubscription }),
    }
}

/// `--spec-backend`, `--spec-draft-tokens` and `--spec-draft-head` (P5-02,
/// GitHub #150, #307). No backend means off, and then a draft window or a
/// head has nothing to configure, so naming one is refused rather than
/// ignored; `off` refuses them too. `dflash2` requires its window — there is
/// no default to guess — while `mtp` has a measured one. The MTP head
/// proposes from its own logits: there is no second head to choose.
fn derive_speculation(
    resolution: &Resolution,
    spelled: &impl Fn(&str, &str) -> String,
) -> Result<(Option<Speculation>, bool), ConfigError> {
    let s = &resolution.settings.spec;
    let named = |name: &str| resolution.explicit("spec", name);
    let shown = |name: &str, value: String| format!("`{} {value}`", spelled("spec", name));
    let draft_tokens = s.draft_tokens.filter(|_| named("draft_tokens"));
    let draft_head = s.draft_head.filter(|_| named("draft_head"));
    let backend = match s.backend {
        Some(SpecChoice::Off) => {
            for (name, value) in [("draft_tokens", draft_tokens.map(|n| n.to_string())), ("draft_head", draft_head.map(|h| h.as_str().to_owned()))] {
                if let Some(value) = value {
                    return Err(ConfigError(format!(
                        "{} has nothing to configure under `--spec-backend off`",
                        shown(name, value)
                    )));
                }
            }
            return Ok((None, true));
        }
        None => {
            for (name, value) in [("draft_head", draft_head.map(|h| h.as_str().to_owned())), ("draft_tokens", draft_tokens.map(|n| n.to_string()))] {
                if let Some(value) = value {
                    return Err(ConfigError(format!(
                        "{} requires `--spec-backend` (speculation is off without it)",
                        shown(name, value)
                    )));
                }
            }
            return Ok((None, false));
        }
        Some(SpecChoice::Backend(backend)) => backend,
    };
    let head = s.draft_head.unwrap_or_default();
    if backend == SpeculativeBackend::Mtp && head != ProposalHead::Full {
        return Err(ConfigError(format!(
            "`{}`: `--spec-backend mtp` has no proposal head to choose",
            spelled("spec", "draft_head")
        )));
    }
    let tokens = match s.draft_tokens {
        Some(tokens) => tokens,
        // GitHub #307: Flash-Next's head has a measured default window.
        None if backend == SpeculativeBackend::Mtp => FLASH_NEXT_DEFAULT_DRAFT_TOKENS,
        None => {
            return Err(ConfigError(format!(
                "`{} {}` requires `--spec-draft-tokens N` (N in 1..={MAX_DRAFT_TOKENS})",
                spelled("spec", "backend"),
                backend.as_str()
            )));
        }
    };
    let speculation = Speculation::new(backend, tokens)
        .map_err(|_| ConfigError(format!("`{}` must be in 1..={MAX_DRAFT_TOKENS}, got {tokens}", spelled("spec", "draft_tokens"))))?;
    Ok((Some(speculation.with_proposal_head(head)), false))
}

/// `--vision-enabled`, `--vision-max-tokens` and
/// `--vision-embedding-pool-mib` (GitHub #177, #243). Vision is off unless
/// asked for; neither of the other two has anything to size with it off, so
/// naming one is refused rather than ignored. With vision on, the envelope
/// defaults to [`DEFAULT_VISION_MAX_TOKENS`] and the pool to one
/// envelope-wide embedding — exactly what GitHub #177 always reserved, so a
/// load that says nothing does not move the VRAM plan. The pool's floor is
/// the leaf's: a pool below one envelope-wide embedding is raised there
/// rather than refused here.
fn derive_vision(resolution: &Resolution, spelled: &impl Fn(&str, &str) -> String) -> Result<Option<Vision>, ConfigError> {
    let s = &resolution.settings.vision;
    if !s.enabled {
        for (name, value) in [
            ("max_tokens", resolution.explicit("vision", "max_tokens").then(|| s.max_tokens.to_string())),
            ("embedding_pool_mib", s.embedding_pool_mib.filter(|_| resolution.explicit("vision", "embedding_pool_mib")).map(|n| n.to_string())),
        ] {
            if let Some(value) = value {
                return Err(ConfigError(format!(
                    "`{} {value}` requires `--vision-enabled` (vision is off without it)",
                    spelled("vision", name)
                )));
            }
        }
        return Ok(None);
    }
    let vision = Vision::new(s.max_tokens).map_err(|_| {
        ConfigError(format!(
            "`{}` must be in 1..={VISION_MAX_TOKENS_LIMIT}, got {}",
            spelled("vision", "max_tokens"),
            s.max_tokens
        ))
    })?;
    Ok(Some(match s.embedding_pool_mib {
        Some(mib) => vision.with_pool_bytes(mib * 1024 * 1024),
        None => vision,
    }))
}

/// `--media-allow-private-network` and `--media-cache-mib` (GitHub #179).
/// Without vision there is no media to acquire, so turning the private
/// network on or sizing the cache is refused rather than ignored, as the
/// vision envelope is.
fn derive_media(
    resolution: &Resolution,
    vision: bool,
    spelled: &impl Fn(&str, &str) -> String,
) -> Result<MediaOptions, ConfigError> {
    let s = &resolution.settings.media;
    if !vision {
        if s.allow_private_network && resolution.explicit("media", "allow_private_network") {
            return Err(ConfigError(format!(
                "`{}` requires `--vision-enabled` (vision is off without it)",
                spelled("media", "allow_private_network")
            )));
        }
        if resolution.explicit("media", "cache_mib") {
            return Err(ConfigError(format!(
                "`{} {}` requires `--vision-enabled` (vision is off without it)",
                spelled("media", "cache_mib"),
                s.cache_mib
            )));
        }
        return Ok(MediaOptions::default());
    }
    Ok(MediaOptions { allow_private_network: s.allow_private_network, cache_bytes: s.cache_mib << 20 })
}

/// What an artifact's family decides about a config (spec flash-next/04),
/// once the per-field applicability has been applied by the resolver: the
/// id the load is served under, whether the named backend is this family's
/// drafter, and — Flash-Next attends with its own QSA, not the GQA op — the
/// 27B's attention envelope (GitHub #228).
fn fit_family(
    fitted: FamilyFit<'_>,
    resolution: &Resolution,
    family: ModelFamily,
    fit: Fit,
    spelled: &impl Fn(&str, &str) -> String,
) -> Result<(), ConfigError> {
    let s = &resolution.settings;
    match &s.model.id {
        Some(id) if ModelFamily::of_model_id(id).is_some_and(|named| named != family) => {
            return Err(ConfigError(format!(
                "`{} {id}` names another model than the artifact's, which is {}",
                spelled("model", "id"),
                family.name()
            )));
        }
        Some(_) => {}
        None => *fitted.model = family.model_id().to_owned(),
    }
    if let Some(speculation) = fitted.speculation.filter(|s| s.backend() != family.drafter()) {
        match fit {
            Fit::Start => {
                return Err(ConfigError(format!(
                    "`{} {}`: {} drafts with {}",
                    spelled("spec", "backend"),
                    speculation.backend().as_str(),
                    family.name(),
                    family.drafter().as_str()
                )));
            }
            Fit::Switch => {
                *fitted.speculation = None;
                fitted.dropped.push(schema::field("spec", "backend").expect("declared").flag());
                for name in ["draft_tokens", "draft_head"] {
                    if resolution.explicit("spec", name) {
                        fitted.dropped.push(schema::field("spec", name).expect("declared").flag());
                    }
                }
            }
        }
    }
    let (max_context, kv_format) = (s.model.max_context, s.model.kv_format);
    if family == ModelFamily::Qwen38_27b && max_context > kv_format.gqa_max_context() {
        return Err(ConfigError(format!(
            "`{} {max_context}`: the 27B's attention serves at most {} keys on `{}`",
            spelled("model", "max_context"),
            kv_format.gqa_max_context(),
            kv_format.as_str()
        )));
    }
    Ok(())
}

fn version_text() -> String {
    format!("ignis-server {}", env!("CARGO_PKG_VERSION"))
}

/// `--help`: how to run the server, and every field's flag, env var and
/// default, one line each, from the field table itself.
pub fn help_text() -> String {
    let mut text = String::from(
        "ignis-server: the OpenAI-compatible HTTP entrypoint\n\
         \n\
         USAGE:\n    ignis-server [--config <path>] [--profile <name>] [FIELD FLAGS]\n\
         \x20   ignis-server help [--fields [--format text|json]]\n\
         \x20   ignis-server version\n\
         \x20   ignis-server config generate [--format json|yaml] [--out <path>] [--force] [--dry-run] [FIELD FLAGS]\n\
         \x20   ignis-server config print [--file <path>] [--format json|yaml] [FIELD FLAGS]\n\
         \x20   ignis-server config patch [--file <path>] [--out <path>] [FIELD FLAGS]\n\
         \n\
         Every field has a flag (--<group>-<field>), an env var (IGNIS_<GROUP>_<FIELD>)\n\
         and a config-file key (<group>.<field>). A flag overrides its env var, which\n\
         overrides the built-in default.\n",
    );
    for (group, fields) in schema::GROUPS {
        text.push_str(&format!("\n{}:\n", group.to_ascii_uppercase()));
        for meta in *fields {
            let default = match (meta.default)() {
                serde_json::Value::Null => "unset".to_owned(),
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            text.push_str(&format!(
                "    {} <{}>  env: {}  (default: {default}{})\n        {}\n",
                meta.flag(),
                meta.kind,
                meta.env(),
                match meta.validator.describe() {
                    rule if rule.is_empty() => String::new(),
                    rule => format!("; {rule}"),
                },
                meta.summary()
            ));
        }
    }
    text.push_str("\n    -h, --help        print this help and exit\n    -V, --version     print the version and exit\n");
    text
}
