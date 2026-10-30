//! Every configurable field, each declared once (ADR 0046, spec
//! config-v2/01).
//!
//! One [`config_group!`] invocation per group lists that group's fields; one
//! entry per field expands to both the struct field and its [`FieldMeta`] —
//! the flag, env var, file key, default, validator, applicability, the
//! `GET`/`PATCH /v1/config` attributes and the help text all come from that
//! one written entry, so none of them can drift from the others. One
//! invocation per group rather than per field because `macro_rules!` cannot
//! gather separate invocations into one table; inside it, a field is still
//! written exactly once.
//!
//! A field's entry reads:
//!
//! ```text
//! /// Operator-facing description (rustdoc and `help --fields` alike).
//! name: Kind = default, validator, applicability, [Attr, ...];
//! ```
//!
//! `Kind` is a [`FieldKind`] (`kind.rs`) — it decides the Rust type, the
//! grammar and the written form; the attributes ([`Attr`]) are all off
//! unless listed. The field's flag is `--<group>-<name>`, its env var
//! `IGNIS_<GROUP>_<NAME>`, its file key `<group>.<name>`.
//!
//! Names follow today's flags, with the group's own word dropped where it
//! would repeat (`--vram-headroom-bytes`, not `--vram-vram-headroom-bytes`;
//! `--reuse-prompt`, not `--reuse-prompt-reuse`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use ignis_core::compute::ModelFamily;
use ignis_core::ngram_cache::CacheLocation;
use ignis_core::{ExceptionWeights, KvFormat, PrefillAttention, RopeSetting};
use serde_json::{Map, Value};

use super::field::{Applicability, Attr, FieldMeta, Validator};
use super::kind::*;
use super::source::Resolver;
use super::ConfigError;
use crate::instruction::{DeveloperMessagePolicy, SystemMessagePolicy};

/// Expand one group's field list into its settings struct, its `FIELDS`
/// table, and the resolve/render functions the rest of the module reads
/// them through. See the module doc for the entry syntax.
macro_rules! config_group {
    (
        $(#[doc = $gdoc:literal])*
        $Group:ident = $gname:literal {
            $(
                $(#[doc = $doc:literal])*
                $field:ident: $kind:ty = $default:expr, $validator:expr, $applies:expr $(, [$($attr:ident),* $(,)?])?;
            )*
        }
    ) => {
        $(#[doc = $gdoc])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $Group {
            $(
                $(#[doc = $doc])*
                pub $field: <$kind as FieldKind>::Value,
            )*
        }

        impl Default for $Group {
            /// Every field at its hardcoded default.
            fn default() -> Self {
                Self { $( $field: $default, )* }
            }
        }

        impl $Group {
            /// The group's name, as the file key and the env var spell it.
            pub const NAME: &'static str = $gname;

            /// One entry per struct field, in declaration order.
            pub const FIELDS: &'static [FieldMeta] = &[
                $(
                    FieldMeta {
                        group: $gname,
                        name: stringify!($field),
                        kind: <$kind as FieldKind>::TAG,
                        nullable: <$kind as FieldKind>::NULLABLE,
                        default: || <$kind as FieldKind>::render(&$default),
                        file_text: <$kind as FieldKind>::file_text,
                        canonical: |raw| <$kind as FieldKind>::parse(raw).map(|value| <$kind as FieldKind>::render(&value)),
                        description: concat!($($doc, "\n",)*),
                        validator: $validator,
                        applies: $applies,
                        visible: Attr::listed(&[$($(Attr::$attr),*)?], Attr::Visible),
                        patchable: Attr::listed(&[$($(Attr::$attr),*)?], Attr::Patchable),
                        reload_required: Attr::listed(&[$($(Attr::$attr),*)?], Attr::ReloadRequired),
                        scoped: Attr::listed(&[$($(Attr::$attr),*)?], Attr::Scoped),
                        switch: <$kind as FieldKind>::SWITCH,
                        repeatable: <$kind as FieldKind>::REPEATABLE,
                    },
                )*
            ];

            /// The entry for the field called `name`.
            fn meta(name: &str) -> &'static FieldMeta {
                Self::FIELDS.iter().find(|meta| meta.name == name).expect("a field the macro declared")
            }

            /// Every field resolved through `resolver`'s sources, in
            /// declaration order (the first refusal wins, so it is the
            /// first field written here that is wrong).
            pub(super) fn resolve(resolver: &Resolver<'_>) -> Result<Self, ConfigError> {
                Ok(Self {
                    $( $field: resolver.field::<$kind>(Self::meta(stringify!($field)), $default)?, )*
                })
            }

            /// Every field as the file format writes it.
            pub fn render(&self) -> Map<String, Value> {
                let mut fields = Map::new();
                $( fields.insert(stringify!($field).to_owned(), <$kind as FieldKind>::render(&self.$field)); )*
                fields
            }
        }
    };
}

// The tests that pin the attributes' closed defaults declare throwaway
// groups with it, beside the fields they test.
#[cfg(test)]
pub(crate) use config_group;

/// Every group, in the order `help --fields`, a written file and the
/// resolver visit them: builds [`Settings`], [`GROUPS`] and their two
/// whole-config functions from one list, as `config_group!` does per field.
macro_rules! settings {
    ($( $(#[doc = $doc:literal])* $group:ident: $Group:ident, )*) => {
        /// The resolved value of every declared field, grouped as the config
        /// file nests them — spec config-v2/01's "resolved Config". What `GET
        /// /v1/config` shows (filtered to [`FieldMeta::visible`]), what
        /// `config generate` writes, and what [`super::Config`] is derived from.
        #[derive(Debug, Clone, PartialEq, Eq, Default)]
        pub struct Settings {
            $( $(#[doc = $doc])* pub $group: $Group, )*
        }

        /// Every group's name and field table, in declaration order.
        pub const GROUPS: &[(&str, &[FieldMeta])] = &[ $( ($Group::NAME, $Group::FIELDS), )* ];

        impl Settings {
            /// Every group resolved through `resolver`.
            pub(super) fn resolve(resolver: &Resolver<'_>) -> Result<Self, ConfigError> {
                Ok(Self { $( $group: $Group::resolve(resolver)?, )* })
            }

            /// Every field, nested by group, as the file format writes it.
            pub fn render(&self) -> Map<String, Value> {
                let mut groups = Map::new();
                $( groups.insert($Group::NAME.to_owned(), Value::Object(self.$group.render())); )*
                groups
            }
        }

        impl serde::Serialize for Settings {
            /// Nested by group, every field as the file format writes it.
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serde::Serialize::serialize(&Value::Object(self.render()), serializer)
            }
        }

        impl<'de> serde::Deserialize<'de> for Settings {
            /// A whole config document, read as a file is — so an unknown
            /// field or a bad value is refused by name — over the hardcoded
            /// defaults.
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
                settings_from_document(&value).map_err(serde::de::Error::custom)
            }
        }

        $(
            impl serde::Serialize for $Group {
                /// Every field as the file format writes it.
                fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    serde::Serialize::serialize(&Value::Object(self.render()), serializer)
                }
            }

            impl<'de> serde::Deserialize<'de> for $Group {
                /// The group's own section of a config document, read as
                /// [`Settings`] reads a whole one.
                fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                    let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
                    let mut document = Map::new();
                    document.insert($Group::NAME.to_owned(), value);
                    settings_from_document(&Value::Object(document))
                        .map(|settings| settings.$group)
                        .map_err(serde::de::Error::custom)
                }
            }
        )*
    };
}

/// A config document read on its own over the hardcoded defaults: through
/// the file reader and the resolver, the one path every value takes, so a
/// group's `Deserialize` cannot read a field differently from a start. The
/// general values only — a family section waits for a family.
fn settings_from_document(value: &Value) -> Result<Settings, String> {
    let document = super::file::read_document(value, "the document").map_err(|e| e.0)?;
    let sources = super::source::Sources { file: document.values, ..Default::default() };
    super::source::resolve_settings(&sources, None, super::source::Fit::Start)
        .map(|resolution| resolution.settings)
        .map_err(|e| e.0)
}

/// Every field of every group, in declaration order.
pub fn all_fields() -> impl Iterator<Item = &'static FieldMeta> {
    GROUPS.iter().flat_map(|(_, fields)| fields.iter())
}

/// The field `<group>.<name>`, if one is declared.
pub fn field(group: &str, name: &str) -> Option<&'static FieldMeta> {
    all_fields().find(|meta| meta.group == group && meta.name == name)
}

const ALL: Applicability = Applicability::AllFamilies;
const FLASH_NEXT_ONLY: Applicability = Applicability::Only(&[ModelFamily::FlashNext]);
const QWEN38_ONLY: Applicability = Applicability::Only(&[ModelFamily::Qwen38_27b]);
const NEMOTRON_DIARIZATION_ONLY: Applicability = Applicability::Only(&[ModelFamily::Nemotron3Diarization]);
const NO_RULE: Validator = Validator::None;

const fn between(min: i64, max: i64) -> Validator {
    Validator::Range { min: Some(min), max: Some(max) }
}

const fn at_least(min: i64) -> Validator {
    Validator::Range { min: Some(min), max: None }
}

/// A decimal field's range, in its own units: `decimal(0.0, 4.0)`.
const fn decimal(min: f64, max: f64) -> Validator {
    Validator::DecimalRange { min: (min * 1000.0) as i64, max: (max * 1000.0) as i64 }
}

config_group! {
    /// The HTTP server itself: where it listens, who may call it, what else
    /// it serves beside `/v1`.
    ServerGroup = "server" {
        /// The API's listen address (host:port). Localhost on the OpenAI port
        /// by default; never changed at runtime — a rebind is a restart.
        bind: Text = super::DEFAULT_BIND.to_owned(), NO_RULE, ALL, [Visible];
        /// The key every `/v1` request must present as `Authorization: Bearer
        /// <key>`; `auto` generates one at start and prints it (the only time
        /// a key is printed). Unset keeps the API open, as it has always been
        /// on localhost. Never shown by `GET /v1/config`.
        api_key: Opt<ApiKeyKind> = None, NO_RULE, ALL;
        /// Expose the server beyond its bind address: `cloudflare-quick` opens
        /// a public https://*.trycloudflare.com URL, printed at start. An
        /// exposed server always requires an API key — `auto` when none is
        /// set.
        expose: Opt<ExposeKind> = None, Validator::OneOf(super::Expose::NAMES), ALL, [Visible];
        /// Serve Prometheus metrics on their own listener (`metrics_bind`),
        /// and at /ui/metrics unless the Playground is off. On by default;
        /// `false` serves neither.
        metrics: Bool = super::DEFAULT_METRICS, NO_RULE, ALL, [Visible];
        /// The metrics listener's address: loopback by default, no API key,
        /// never exposed, never the API's `bind`; refused with `metrics` off.
        /// The default address taken (a second server on the host) is a
        /// warning, and the server starts without the listener; another
        /// address that cannot be bound refuses the start.
        metrics_bind: Text = super::DEFAULT_METRICS_BIND.to_owned(), NO_RULE, ALL, [Visible];
        /// Serve the Playground at /ui/. A binary built without web/dist
        /// serves a page saying how to build it.
        ui: Bool = super::DEFAULT_UI, NO_RULE, ALL, [Visible];
        /// Seconds a request answered whole may run, from its admission (an
        /// image job's from its start), before its deadline ends it as
        /// `timeout_reply` says: chat completions, the Responses API over
        /// HTTP, each question of a decision, an image job, and a request's
        /// media acquisition. 0 is no deadline.
        request_timeout: Secs = super::DEFAULT_REQUEST_TIMEOUT_SECS, between(0, super::MAX_REQUEST_TIMEOUT_SECS as i64), ALL, [Visible, Patchable];
        /// Seconds a request that streams may run, from its admission (an
        /// image job's from its start), before its deadline ends it as
        /// `timeout_reply` says: chat completions and the Responses API
        /// streamed, a Responses WebSocket's response, a streamed image job.
        /// 0 is no deadline.
        stream_timeout: Secs = super::DEFAULT_STREAM_TIMEOUT_SECS, between(0, super::MAX_REQUEST_TIMEOUT_SECS as i64), ALL, [Visible, Patchable];
        /// What a request past its deadline is answered: `partial`, what was
        /// generated by then as an answer a length limit cut
        /// (`finish_reason: "length"`, or a response `incomplete` with
        /// `max_output_tokens`); `error`, a 504 `request_timeout` or a
        /// stream's error ending. A decision's question and an image job are
        /// an error either way.
        timeout_reply: TimeoutReplyKind = crate::deadline::TimeoutReply::Partial, Validator::OneOf(&["partial", "error"]), ALL,
            [Visible, Patchable];
        /// Seconds a Responses WebSocket may sit idle — no frame from the
        /// client and no response running or queued on it — before the
        /// server closes it with a normal close frame saying why; the
        /// client's next request opens another. An open socket holds a
        /// Knative pod, and its GPU, from scaling down. 0 never closes.
        socket_idle_close: Secs = super::DEFAULT_SOCKET_IDLE_CLOSE_SECS, between(0, super::MAX_SOCKET_IDLE_CLOSE_SECS as i64), ALL,
            [Visible, Patchable];
        /// Where `system` messages go: `merge` joins a leading run into the
        /// system prompt and keeps a later one as its own block in place;
        /// `strict` answers 400 for a system message that is not first.
        system_message_policy: SystemPolicy = SystemMessagePolicy::Merge, Validator::OneOf(&["merge", "strict"]), ALL, [Visible, Patchable];
        /// Where `developer` messages go: `inplace`, `into-system`,
        /// `after-system`, `one-after-system` or `reject`. A leading developer
        /// message is the system prompt except under `reject`.
        developer_message_policy: DeveloperPolicy = DeveloperMessagePolicy::Inplace,
            Validator::OneOf(&["inplace", "into-system", "after-system", "one-after-system", "reject"]), ALL, [Visible, Patchable];
    }
}

config_group! {
    /// The model being served and the shape of its load: context, KV
    /// format, prefill, thinking defaults, the rotary table.
    ModelGroup = "model" {
        /// The id the model is served under. Unset, a load is served under
        /// its own model's id (qwen3.8-27b or qwen3.8-flash-next); without an
        /// artifact, qwen3.8-27b. An id naming the other model than the
        /// artifact's is refused.
        id: Opt<Text> = None, NO_RULE, ALL, [Visible];
        /// The .ninfer artifact to load. Unset, the model is looked for under
        /// `download.path`, and fetched when it is not there.
        artifact: Opt<Path> = None, NO_RULE, ALL, [Visible];
        /// The largest prompt + generation a single request may reserve, in
        /// tokens. On the 27B, bounded by its attention's envelope for the KV
        /// format.
        max_context: Tokens = super::DEFAULT_MAX_CONTEXT, at_least(1), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The max_tokens of a request that sends none, its reasoning
        /// included, never past what its prompt leaves of `max_context`; an
        /// explicit cap always wins. 0 = none, up to the context.
        default_max_tokens: Tokens = super::DEFAULT_MAX_TOKENS, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The KV storage format: `hq-e8-2b` (the serving default) or `bf16`
        /// (the retained oracle format).
        kv_format: KvFormatKind = KvFormat::default(), Validator::OneOf(&["bf16", "hq-e8-2b", "hq_e8_2b", "hq"]), ALL,
            [Visible, Patchable, ReloadRequired, Scoped];
        /// The 27B's prompt attention precision (GitHub #335): `fp8` (the
        /// default, the band pass in 8 bits) or `fp16` (the FP16-accumulate
        /// fallback). Flash-Next's attention is QSA and has no such pass.
        prefill_attention: PrefillAttentionKind = PrefillAttention::default(), Validator::OneOf(&PrefillAttention::NAMES), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// How the 27B holds its nine BF16 exception tensors on the device
        /// (GitHub #341): `bf16` (the default, as the artifact stores them)
        /// or `fp8`, converted at load to FP8 row-scale for the FP8 tensor
        /// cores. Flash-Next has none.
        exception_weights: ExceptionWeightsKind = ExceptionWeights::default(), Validator::OneOf(&ExceptionWeights::NAMES), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// The prefill chunk width, in tokens.
        prefill_chunk: Tokens = super::DEFAULT_PREFILL_CHUNK, Validator::MultipleOf(super::PREFILL_CHUNK_ALIGNMENT as u64), ALL,
            [Visible, Patchable, ReloadRequired, Scoped];
        /// The percent of the model's time decoding lanes keep while a prompt
        /// prefills. Unset, the model's own: 25 on both. 0 is one decode
        /// round per chunk, 50 splits time evenly.
        decode_share: Opt<Percent> = None, between(0, 99), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Whether a request that says nothing thinks.
        enable_thinking: Bool = true, NO_RULE, ALL, [Visible, Patchable];
        /// The reasoning_effort of a request that names none. Unset, the
        /// template's own default.
        reasoning_effort: Opt<Effort> = None, Validator::OneOf(&["none", "minimal", "low", "medium", "high", "xhigh", "max"]), ALL,
            [Visible, Patchable];
        /// The reasoning tokens a request may spend before the model's close
        /// is forced; `off` = no budget. A request's own thinking_budget
        /// overrides it.
        thinking_budget: ThinkingBudget = Some(super::DEFAULT_THINKING_BUDGET), NO_RULE, ALL, [Visible, Patchable];
        /// The loop guard (ADR 0052): `rewind`, a loop in a request's
        /// reasoning (chat completions, `/v1/responses`) is taken back and
        /// the model nudged, then closed after `loop_guard_retries` nudges;
        /// `log`, it is only logged; `off`, no guard, as benches and gates
        /// run for reference parity.
        loop_guard: LoopGuard = crate::loop_guard::LoopGuardMode::Rewind, Validator::OneOf(crate::loop_guard::LoopGuardMode::NAMES), ALL,
            [Visible, Patchable];
        /// A window of scored reasoning whose zlib ratio is under this many
        /// thousandths is a loop.
        loop_guard_ratio: Permille = crate::loop_guard::DEFAULT_RATIO_PERMILLE, between(1, 999), ALL, [Visible, Patchable];
        /// The loop guard's window, in characters of scored reasoning.
        loop_guard_window: Chars = crate::loop_guard::DEFAULT_WINDOW_CHARS, between(500, 100_000), ALL, [Visible, Patchable];
        /// Whether the loop guard scores the reasoning normalized: fenced
        /// and inline code taken out, every run of digits read as one `0`.
        /// `off` scores it as written, which at this window and ratio
        /// flags far more than the raw prototype did (3,000 and 120).
        loop_guard_normalize: Bool = crate::loop_guard::DEFAULT_NORMALIZE, NO_RULE, ALL, [Visible, Patchable];
        /// The loop guard's raw gate, in thousandths: a loop's last
        /// `loop_guard_window` characters as written must score under it
        /// too, which a table of numbers (a template once its digits fold)
        /// does not. 0, no gate.
        loop_guard_raw_gate: Permille = crate::loop_guard::DEFAULT_RAW_GATE_PERMILLE, between(0, 999), ALL, [Visible, Patchable];
        /// A window of scored reasoning at or over this many thousandths is
        /// clean: a rewind goes back to the start of the last one before
        /// the loop (no clean one since the last nudge: to its end).
        loop_guard_clean: Permille = crate::loop_guard::DEFAULT_CLEAN_PERMILLE, between(1, 999), ALL, [Visible, Patchable];
        /// How many characters before that clean window's start a rewind
        /// cuts.
        loop_guard_margin: Chars = crate::loop_guard::DEFAULT_MARGIN_CHARS, between(0, 100_000), ALL, [Visible, Patchable];
        /// The nudges a looping reasoning gets before the guard closes it.
        loop_guard_retries: Retries = crate::loop_guard::DEFAULT_RETRIES, between(0, 10), ALL, [Visible, Patchable];
        /// How many checks in a row must score a loop before the guard
        /// acts: a loop persists, a list or a log transcribed into the
        /// reasoning ends with its content. 1, at the first.
        loop_guard_persist: Checks = crate::loop_guard::DEFAULT_PERSIST_CHECKS, between(1, 100), ALL, [Visible, Patchable];
        /// A rewind that would keep fewer characters of reasoning than this
        /// is only logged, and that run of loop scores is not acted on: a
        /// reasoning that opens in a loop, or in an enumeration scored as
        /// one, would otherwise be dropped whole. 0, always rewind.
        loop_guard_min_kept: Chars = crate::loop_guard::DEFAULT_MIN_KEPT_CHARS, between(0, 100_000), ALL, [Visible, Patchable];
        /// In `rewind` mode, the characters of reasoning held back from the
        /// client: a rewind whose cut falls in them reaches it with no tag
        /// and none of the loop. 0, none; under ~2,000 it hides almost no
        /// rewind.
        loop_guard_holdback: Chars = crate::loop_guard::DEFAULT_HOLDBACK_CHARS, between(0, 100_000), ALL, [Visible, Patchable];
        /// The loop guard scores the window again after this many
        /// characters of new reasoning.
        loop_guard_check_every: Chars = crate::loop_guard::DEFAULT_CHECK_EVERY_CHARS, between(1, 100_000), ALL, [Visible, Patchable];
        /// A rewind's cut moves back to the line break before it when one
        /// is within this many characters. 0, never.
        loop_guard_snap: Chars = crate::loop_guard::DEFAULT_SNAP_CHARS, between(0, 100_000), ALL, [Visible, Patchable];
        /// The least of a request's `max_tokens` a continuation must have
        /// left after the cut and the nudge: with less, the loop is only
        /// logged.
        loop_guard_min_continuation: Tokens = crate::loop_guard::DEFAULT_MIN_CONTINUATION_TOKENS, at_least(1), ALL, [Visible, Patchable];
        /// How long a continuation the engine refuses as full waits for
        /// room, in seconds, before the loop is left as it is; 0 tries
        /// once. A WebSocket request's continuation waits in the socket's
        /// admission queue instead, as the socket's own requests do.
        loop_guard_submit_wait: Secs = crate::loop_guard::DEFAULT_SUBMIT_WAIT_SECS, between(0, 3600), ALL, [Visible, Patchable];
        /// The sentence, in the model's own voice, the loop guard's first
        /// rewind puts after the kept reasoning, on a paragraph of its own.
        loop_guard_nudge: Text = crate::loop_guard::DEFAULT_NUDGE.to_owned(), NO_RULE, ALL, [Visible, Patchable];
        /// The sentence every later rewind puts there, until
        /// `loop_guard_retries` nudges close the reasoning.
        loop_guard_nudge_again: Text = crate::loop_guard::DEFAULT_NUDGE_AGAIN.to_owned(), NO_RULE, ALL, [Visible, Patchable];
        /// The text rotary table: `auto` (the default), `none` (the linear
        /// table) or `yarn:F[,t=..][,bf=..][,bs=..]`, which rescales the
        /// checkpoint's trained 262,144-position envelope — what a context
        /// past it needs. `auto` decides per model from its
        /// `model.max_context`: none up to 262,144, else YaRN at
        /// `max_context / 262,144`. Both models run the table.
        rope_scaling: Rope = RopeSetting::Auto, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Seconds with no request using the model after which it is unloaded
        /// — VRAM, pinned host memory and KV-RAM given back — while the server
        /// stays up; the next request loads it again and waits for it. 0 never
        /// unloads.
        idle_unload: Secs = super::DEFAULT_MODEL_IDLE_UNLOAD_SECS, between(0, super::MAX_MODEL_IDLE_UNLOAD_SECS as i64), ALL,
            [Visible, Patchable];
        /// What a text model's host plan must leave of the physical memory
        /// available at load (free plus standby): a load whose pinned lines
        /// — the 27B's token embedding, Flash-Next's experts and n-gram hot
        /// rows, staging, retained host slots, the KV-RAM arena — leave less
        /// is refused, naming the line to shrink, so the desktop is never
        /// paged out. The image model makes no host plan: its start carries
        /// it for a switch to a text model.
        host_margin_bytes: Bytes = super::DEFAULT_HOST_MARGIN_BYTES, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The 27B's adaptation (ADR 0063): when its VRAM plan refuses a
        /// `max_context` the profile or the default gave, the load lowers it
        /// this many tokens at a time, never below this many -- a start whose
        /// plan does not hold one step is refused. A whole number of KV pages
        /// (64 tokens).
        adapt_context_step: Tokens = super::DEFAULT_ADAPT_CONTEXT_STEP, Validator::MultipleOf(ignis_core::KV_PAGE_TOKENS as u64),
            QWEN38_ONLY, [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// The device memory a load may hold, and how the KV pool takes its
    /// share of it.
    VramGroup = "vram" {
        /// The KV pool: a byte count (K/M/G suffix read) or tokens as
        /// `<n>tok`, `<n>Ktok`, `<n>Mtok`. Unset, the KV pool policy's: the
        /// rest of the VRAM budget when every weight is on the device (the
        /// 27B); 524,288 tokens shared by the lanes when Flash-Next's experts
        /// stream, the rest to the expert cache. Refused below one
        /// `max_context` sequence and a page per retained slot.
        kv_pool_bytes: Opt<KvPool> = None, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// What a derived VRAM budget leaves to the desktop and every other
        /// process on the card: the budget is the memory free at start less
        /// this. Not with `budget_bytes`.
        headroom_bytes: Bytes = super::DEFAULT_VRAM_HEADROOM_BYTES, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The device memory the whole process may hold, weights included;
        /// refused above free memory. Unset, derived from `headroom_bytes`.
        budget_bytes: Opt<Bytes> = None, at_least(1), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Start above free memory, with a warning. Needs `budget_bytes`.
        allow_oversubscription: Bool = false, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Start Flash-Next with an expert cache below its 12 GiB floor, with
        /// a warning — decode slows sharply below it.
        allow_expert_cache_below_floor: Bool = false, NO_RULE, FLASH_NEXT_ONLY, [Visible, Patchable, ReloadRequired];
        /// Flash-Next's calibration-hottest experts held in the expert cache
        /// for the whole load, outside its LRU, with no copy in the pinned
        /// host pool: whole experts, hottest first, until they reach this many
        /// bytes. The pool takes that much less RAM; the LRU has those slots
        /// fewer (on the 5090: nothing measurable at 1 GiB, ~1% of three-lane
        /// decode at 2 GiB). 0 pins none; the `rtx5090` profile pins 2 GiB. A
        /// card that holds every expert keeps none in RAM anyway and ignores it.
        pinned_expert_bytes: Bytes = 0, NO_RULE, FLASH_NEXT_ONLY, [Visible, Patchable, ReloadRequired];
        /// The 27B's adaptation (ADR 0063): at more than one decode lane, a
        /// context the load lowers is kept only when the KV pool holds it and
        /// a spare reservation of this many tokens, room for a second request;
        /// 0, one sequence is enough. At one lane the plan's own floor decides.
        adapt_spare_tokens: Tokens = super::DEFAULT_ADAPT_SPARE_TOKENS, NO_RULE, QWEN38_ONLY, [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// Prompt reuse across requests and the KV-RAM host tier it keeps state
    /// in (the arena is prompt reuse's, not the VRAM pool's).
    ReuseGroup = "reuse" {
        /// Cross-request state reuse. Off, no prompt checkpoint is captured
        /// or claimed, and no prefix is shared unless `retained_device` or
        /// `retained_host` gives slots for it.
        prompt: Bool = super::DEFAULT_PROMPT_REUSE, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Retained slots in VRAM, reserved in the VRAM plan and handed out
        /// first. Unset, 0. With prompt reuse off, a count shares heads
        /// between live siblings only.
        retained_device: Opt<Slots> = None, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Retained slots in one pinned host block reserved at start, ~222
        /// MiB each, a PCIe copy per capture and per claim. Unset, the
        /// model's own: two per decode lane on the 27B, 8 on Flash-Next; 0
        /// with prompt reuse off.
        retained_host: Opt<Slots> = None, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Idle seconds after which a main-conversation checkpoint in KV-RAM
        /// ranks as a subagent's. Needs prompt reuse on.
        retained_interactive_ttl: Secs = super::DEFAULT_RETAINED_INTERACTIVE_TTL_SECS, NO_RULE, ALL,
            [Visible, Patchable, ReloadRequired, Scoped];
        /// The KV-RAM host tier's budget: page-locked whole at start and held
        /// for the life of the load, so it is RAM the process holds even idle
        /// (and the figure Windows reports as its shared GPU memory). 0
        /// disables the tier.
        kv_host_pool_bytes: Bytes = super::DEFAULT_HOST_POOL_BYTES, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
    }
}

config_group! {
    /// The runtime model switch (spec model-switch/01): how a switch drains,
    /// and which models a request may switch to by naming them.
    SwitchGroup = "switch" {
        /// How long a model switch lets the old model's running requests
        /// finish before it cancels them, in seconds; 0 cancels them at once.
        drain_timeout: Secs = super::DEFAULT_SWITCH_DRAIN_TIMEOUT_SECS, between(0, super::MAX_SWITCH_DRAIN_TIMEOUT_SECS as i64), ALL,
            [Visible, Patchable];
        /// Whether a request whose model names another model `known_models`
        /// lists switches the server to it, then is served on it. Off refuses
        /// it as an unknown model is. An explicit switch is not affected.
        allow_implicit: Bool = super::DEFAULT_ALLOW_MODEL_SWITCH, NO_RULE, ALL, [Visible, Patchable];
        /// The models a request may switch to by naming them, each with the
        /// artifact it loads from: `<id>=<path>` (the flag is repeatable, the
        /// env var takes `;`-separated pairs, a file a map). The model the
        /// server starts on joins them once its load says its id.
        known_models: KnownModels = BTreeMap::new(), NO_RULE, ALL, [Visible, Patchable];
    }
}

config_group! {
    /// Speculative decoding, and the decode lanes.
    SpecGroup = "spec" {
        /// The speculative backend: `dflash2` (the 27B's drafter), `mtp`
        /// (Flash-Next's head, with its companion container beside the
        /// artifact), or `off`. Unset, no speculation.
        backend: Opt<SpecBackend> = None, Validator::OneOf(&["dflash2", "mtp", "off"]), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The draft window: required with `dflash2`; with `mtp`, the most
        /// drafts a lane verifies (default 2). Needs a backend.
        draft_tokens: Opt<Tokens> = None, between(1, ignis_core::MAX_DRAFT_TOKENS as i64), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Flash-Next's draft row budget: the rows a verify round takes across
        /// lanes, 0 (= 8) or 2..=32; past 8, five to eight lanes draft too.
        draft_rows: Opt<DraftRows> = None, between(0, ignis_core::speculation::FLASH_NEXT_MAX_VERIFY_ROWS as i64), FLASH_NEXT_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// The decode lanes: the sequences a model decodes at once. On the
        /// 27B each lane reserves its state (~233 MB with `dflash2` under
        /// hq-e8-2b) and its share of the round buffers, and takes two of the
        /// default host retained slots. On Flash-Next they decode sharing the
        /// KV pool — min(524,288 tokens, lanes x `max_context`), never below
        /// one `max_context` and a page per retained slot. Unset, the model's
        /// own: 8 on the 27B, 3 on Flash-Next.
        decode_lanes: Opt<Lanes> = None, between(1, ignis_core::N_DECODE_LANES as i64), ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// The drafter's proposal head: `full` (the target's output head),
        /// `shortlist` (the artifact's Q4 head over the 131,072 most frequent
        /// tokens, ~357 MB more VRAM) or `auto` (`full` on every card class: the
        /// shortlist frees no VRAM and measured slower). Needs a backend; `mtp`
        /// has only `full`. Unset, `full`.
        draft_head: Opt<DraftHead> = None, Validator::OneOf(&["full", "shortlist", "auto"]), ALL, [Visible, Patchable, ReloadRequired, Scoped];
    }
}

config_group! {
    /// The 27B's vision tower.
    VisionGroup = "vision" {
        /// Load the vision tower and reserve its workspace.
        enabled: Bool = false, NO_RULE, QWEN38_ONLY, [Visible, Patchable, ReloadRequired];
        /// Merged vision tokens per request. An image over it is shrunk to
        /// fit, aspect kept; refused only when its aspect ratio cannot fit, or
        /// when several images total over it. Needs vision on.
        max_tokens: Tokens = ignis_core::DEFAULT_VISION_MAX_TOKENS, between(1, ignis_core::VISION_MAX_TOKENS_LIMIT as i64), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// Encoded images kept for reuse, in MiB, where `vision.residency`
        /// puts the pool: VRAM, or pinned host RAM under `host`. Unset, one
        /// envelope-wide embedding; a pool below that is raised to it. Needs
        /// vision on.
        embedding_pool_mib: Opt<Mib> = None, between(1, super::MAX_VISION_EMBEDDING_POOL_MIB as i64), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
        /// Where the vision tower's weights and the embedding pool live:
        /// `device`, in VRAM for the life of the load (today's encode
        /// latency), or `host`, in pinned host RAM, the weights copied to the
        /// GPU for each image encode (~300 MB and the pool less VRAM, so more
        /// KV pool, a few ms more an encode). A `host` start the host RAM
        /// cannot hold is refused, never moved to the device. Needs vision on.
        residency: VisionResidencyKind = ignis_core::VisionResidency::default(), Validator::OneOf(&ignis_core::VisionResidency::NAMES), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// How image parts are acquired.
    MediaGroup = "media" {
        /// Fetch image URLs on private, loopback, link-local, multicast and
        /// CGNAT addresses. Off, so an exposed server cannot be used to probe
        /// the operator's LAN. Needs vision on.
        allow_private_network: Bool = false, NO_RULE, QWEN38_ONLY, [Visible, Patchable, ReloadRequired];
        /// Prepared images kept for reuse, in MiB; 0 keeps none. Needs vision
        /// on.
        cache_mib: Mib = super::DEFAULT_MEDIA_CACHE_MIB as u64, between(0, super::MEDIA_CACHE_MIB_LIMIT as i64), QWEN38_ONLY,
            [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// The image model's serving (spec image-model/05, ADR 0050): its job
    /// queue, what one request may ask for, the condition budget, prompt
    /// enhancement. Read while any model is loaded — a text model is the
    /// enhancer — so every family takes them, and each changes live.
    ImageGroup = "image" {
        /// The image jobs that may wait behind the one running, in order; one
        /// more is refused `503 engine_full` with `Retry-After`. 0 runs one
        /// at a time with none waiting.
        queue_depth: Jobs = 4, between(0, 64), ALL, [Visible, Patchable];
        /// The most images one request may ask for (`n`), made one after
        /// another over the job's one condition prefix.
        max_n: Images = 10, between(1, 64), ALL, [Visible, Patchable];
        /// The most condition images one edit may send. At most 10, what the
        /// image model's encoder holds.
        max_condition_images: Images = 10, between(1, 10), ALL, [Visible, Patchable];
        /// The largest a condition image may decode to, in Mi pixels
        /// (1,048,576 pixels, a 1024x1024 image, each); a larger one is
        /// refused. A condition image is encoded at 1024x1024 of area at
        /// most, so 16 — a 16-megapixel photo, 64 MiB decoded — takes any
        /// phone camera's own output and bounds what one condition image
        /// decodes to.
        max_condition_mipx: Mipx = 16, between(1, 64), ALL, [Visible, Patchable];
        /// The most edits that decode their condition images at once; the
        /// others wait their turn. Each holds up to `max_condition_images`
        /// frames of up to `max_condition_mipx` while it decodes, and the
        /// queue runs one job at a time, so more side by side buy memory,
        /// not throughput.
        max_condition_decodes: Jobs = 1, between(1, 16), ALL, [Visible, Patchable];
        /// The sizes a request may ask for, `<width>x<height>` with each side
        /// a multiple of 32: by default the checkpoint's presets and
        /// 1024x1024, the size OpenAI clients often send. `auto` picks among
        /// them.
        sizes: Sizes = super::DEFAULT_IMAGE_SIZES.to_vec(), NO_RULE, ALL, [Visible, Patchable];
        /// The largest image the image model's load makes room for, in
        /// latent cells of 16x16 pixels: 16,800, 2400x1792, the checkpoint's
        /// largest preset. The load sizes its working memory for the largest
        /// of `sizes` within it; a size past it is not served, and the load
        /// warns. Read by the load, so a change holds from the next one.
        max_target_cells: Cells = ignis_core::image_leaf::MAX_TARGET_TOKENS, between(1, 1 << 20), ALL, [Visible, Patchable];
        /// The condition budget, in condition-prefix tokens (~256 KiB each, a
        /// 1024x1024 condition image 4,096 of them). 0 is the whole budget the
        /// load planned; a value can only lower it.
        condition_budget_tokens: Tokens = 0, NO_RULE, ALL, [Visible, Patchable];
        /// The least condition budget the image model's load accepts, in
        /// condition-prefix tokens: a load whose VRAM plan leaves less is
        /// refused, naming the shortfall, rather than refuse its requests one
        /// by one. 8,191, the longest text-to-image prompt the image model's
        /// positions admit; 0 takes any. Read by the load, so a change holds
        /// from the next one.
        min_condition_budget_tokens: Tokens = ignis_core::image_leaf::DEFAULT_MIN_CONDITION_BUDGET_TOKENS, between(0, 1 << 20), ALL,
            [Visible, Patchable];
        /// The smallest the budget may scale a condition image down to, as
        /// the side of a square of the same area, in pixels; a request whose
        /// images would go below it is refused, naming the budget.
        condition_floor_px: Pixels = 512, between(32, 1024), ALL, [Visible, Patchable];
        /// Whether a request that does not say has its prompt rewritten by
        /// `prompt_enhancer` before the image model reads it.
        prompt_enhancement: Bool = false, NO_RULE, ALL, [Visible, Patchable];
        /// The text model that rewrites prompts: the served id of a model an
        /// explicit switch may name (`switch.known_models`, the start model,
        /// or a downloaded catalog entry), whatever `switch.allow_implicit`
        /// says, or the text model loaded. A `PATCH` naming any other is
        /// refused, and a start warns of it; a request that meets one anyway
        /// skips the rewrite, with a warning.
        prompt_enhancer: Text = ModelFamily::Qwen38_27b.model_id().to_owned(), NO_RULE, ALL, [Visible, Patchable];
        /// The most tokens one rewrite may take. Qwen's prompts ask for
        /// about 200 words; a longer answer is cut here and used as it is.
        prompt_enhancer_max_tokens: Tokens = 1024, between(16, 8192), ALL, [Visible, Patchable];
    }
}

config_group! {
    /// The speech model (spec speech/02, ADR 0067): Qwen3-ASR, loaded beside
    /// the active model on the CPU at first use. Read while any model is
    /// loaded, so every family takes them, and each changes live.
    SpeechGroup = "speech" {
        /// The speech model: `auto` (the 1.7B Q4_K_M catalog entry,
        /// `qwen3-asr-1.7b`), a catalog entry's id, or `off`. It is enabled
        /// when its files are on disk; nothing is ever downloaded by itself
        /// (`ignis-server model download <id>`).
        model: Text = super::DEFAULT_SPEECH_MODEL.to_owned(), NO_RULE, ALL, [Visible, Patchable];
        /// The model names OpenAI clients send that the speech model answers
        /// to, so an unmodified client works: Aider hard-codes `whisper-1`.
        /// Not catalog aliases and not listed by `GET /v1/models`; a name
        /// equal to a served id is refused at start.
        compat_names: Names = super::DEFAULT_SPEECH_COMPAT_NAMES.iter().map(|name| (*name).to_owned()).collect(), NO_RULE, ALL,
            [Visible, Patchable];
        /// The threads the speech model computes with: `auto` is the physical
        /// cores less one, counted within the CPUs the server may run on (a
        /// container's cpuset or quota). Read by the load, so a change holds
        /// from the next one; each load logs the count (`ignis.speech.threads`).
        threads: Threads = None, NO_RULE, ALL, [Visible, Patchable];
        /// The longest clip one transcription takes, in seconds; a longer one
        /// is refused and nothing is split. 1,200, the model's limit.
        max_clip_seconds: Secs = 1200, between(1, 1200), ALL, [Visible, Patchable];
        /// The most tokens of context (`prompt` and `keywords[]`) a
        /// transcription reads; over it the last ones are kept. 0 reads none.
        context_max_tokens: Tokens = 512, between(0, 4096), ALL, [Visible, Patchable];
        /// On a pause the model can write a line of its context in place of
        /// what it heard: a transcript that only repeats this many words in a
        /// row of one context line, or a whole line of two words or more, is
        /// taken for no speech and comes back empty. One word alone never is.
        /// 0 turns the guard off. Read when a session opens, so a change
        /// holds from the next one.
        context_copy_words: Words = 3, between(0, 32), ALL, [Visible, Patchable];
        /// Pin the speech model's weights in host memory instead of mapping
        /// the file. Read by the load, so a change holds from the next one.
        pinned_weights: Bool = false, NO_RULE, ALL, [Visible, Patchable];
        /// Seconds with no transcription, and no Realtime session open, after
        /// which the speech model is unloaded, its host memory given back; the
        /// next transcription loads it again. 0 never unloads.
        idle_unload: Secs = super::DEFAULT_MODEL_IDLE_UNLOAD_SECS, between(0, super::MAX_MODEL_IDLE_UNLOAD_SECS as i64), ALL,
            [Visible, Patchable];
        /// How often at most a Realtime session's live transcript is
        /// refreshed while audio arrives, in milliseconds: each refresh sends
        /// the new stable text, and the provisional tail to a session that
        /// opted in. Best effort: a refresh is skipped while the session's
        /// windows are still being computed, and waits longer after a costly
        /// one (`speech.partial_cost_factor`). Read by every open session at
        /// its next refresh.
        partial_interval_ms: Millis = 2000, between(100, 60_000), ALL, [Visible, Patchable];
        /// How long a Realtime session's next refresh waits at least, as a
        /// multiple of the compute time of its last one, from when that one
        /// was queued; never less than `speech.partial_interval_ms`. Each
        /// refresh decodes the item so far again, so it costs more as the
        /// item grows: the refreshes then take at most about one part in this
        /// many of the speech worker. 1 waits only for the last refresh to
        /// finish; 2 and 3 measured no faster commits on a 10-core or a
        /// 4-core CPU. Read by every open session at its next refresh.
        partial_cost_factor: Times = 1, between(1, 16), ALL, [Visible, Patchable];
        /// The tokens of a live transcript held back as provisional, the rest
        /// being stable: the tail a Realtime session that asks for it gets.
        /// Read by the speech executor when a session opens, so a change holds
        /// from the next session.
        rollback_tokens: Tokens = 10, between(0, 64), ALL, [Visible, Patchable];
        /// The first refreshes of a Realtime session's live transcript that
        /// make no text stable, whatever `speech.rollback_tokens` says: the
        /// language and the opening words are decided on more than one
        /// interval of audio (the official streaming's `unfixed_chunk_num`).
        /// 0 fixes text from the first refresh. Read by the speech executor
        /// when a session opens, so a change holds from the next session.
        unfixed_partials: Partials = 2, between(0, 16), ALL, [Visible, Patchable];
        /// The most Realtime transcription sessions open at once; past it
        /// the next is refused at the upgrade `503` with `Retry-After`.
        max_sessions: Sessions = 4, between(1, 64), ALL, [Visible, Patchable];
        /// The batch transcriptions that may wait behind the one running, in
        /// order; one more is refused `503 engine_full` with `Retry-After`.
        /// 0 runs one at a time with none waiting.
        max_queue: Requests = 4, between(0, 64), ALL, [Visible, Patchable];
        /// The speech probability at or over which `server_vad` hears speech
        /// in 32 ms of a Realtime session's audio: the default of its
        /// `threshold`. Read when a session opens.
        vad_threshold: Fraction = 500, NO_RULE, ALL, [Visible, Patchable];
        /// The audio from before the speech that a `server_vad` item keeps,
        /// in milliseconds: the default of its `prefix_padding_ms`. Read when
        /// a session opens.
        vad_prefix_padding_ms: Millis = 300, between(0, 10_000), ALL, [Visible, Patchable];
        /// The silence after the speech at which `server_vad` commits the
        /// item, in milliseconds: the default of its `silence_duration_ms`.
        /// Read when a session opens.
        vad_silence_duration_ms: Millis = 500, between(0, 60_000), ALL, [Visible, Patchable];
        /// The digital silence put before each `server_vad` item's audio
        /// when it is fed to the speech model, in milliseconds: the model's
        /// transcript depends on the audio before the speech, and an item
        /// starts only its prefix padding before it. No event and no `usage`
        /// counts it; it counts toward `speech.max_clip_seconds`, as the model
        /// reads it. A manual commit's item gets none. Read at each item's
        /// start.
        vad_leading_silence_ms: Millis = ignis_speech::vad::DEFAULT_LEADING_SILENCE_MS, between(0, 10_000), ALL, [Visible, Patchable];
        /// How long a loan of free KV pages to the speech model outlives its
        /// transcription, in seconds (spec speech/06), not served yet.
        loan_grace_seconds: Secs = 30, between(0, 3600), ALL, [Visible, Patchable];
        /// How long a transcription waits for a loan before it runs on the
        /// CPU, in seconds (spec speech/06), not served yet.
        loan_wait_seconds: Secs = 5, between(0, 600), ALL, [Visible, Patchable];
        /// The length guard: once a `server_vad` item holds this many seconds
        /// of audio, the server ends it at the next 32 ms the VAD hears under
        /// its threshold, a pause too short to end it, or
        /// `speech.utterance_cut_search_ms` later at the latest, and the
        /// speech goes on in a new item; so a room that never falls silent
        /// loses nothing to `speech.max_clip_seconds`. A session's own
        /// `session.ignis.max_utterance_seconds` overrides it. 0 never cuts.
        /// Read at each item's audio.
        max_utterance_seconds: Secs = 120, between(0, 1200), ALL, [Visible, Patchable];
        /// How long past `speech.max_utterance_seconds` the length guard looks
        /// for a pause, in milliseconds, before it cuts where the audio is.
        /// The search looks forward only, so no text already sent is past the
        /// cut. In fluent speech the VAD hears a pause within 10 s 95 times in
        /// 100 (spec speech/08, As built).
        utterance_cut_search_ms: Millis = 10_000, between(0, 60_000), ALL, [Visible, Patchable];
        /// The audio a Realtime session may hold that is not computed yet, in
        /// seconds: received, and still waiting for the speech model's windows
        /// or the diarizer's steps. An append past it is refused with an
        /// `error`, `ignis_audio_queue_full`, and the session stays open: a
        /// client feeding a file faster than the CPU computes it paces itself
        /// by it, and one fed at real time never reaches it. Read at each
        /// append.
        max_queued_audio_seconds: Secs = 120, between(1, 3600), ALL, [Visible, Patchable];
    }
}

config_group! {
    /// The diarizer (spec speech/08, ADR 0069): Nemotron 3 Diarization, who
    /// spoke when, beside the speech model on the CPU, for the Realtime
    /// sessions that ask for speakers. The geometries and the speaker cache's
    /// constants are Sortformer's, so only its family takes them. A session
    /// reads them when its speakers start, so each changes live from the
    /// next.
    DiarizerGroup = "diarizer" {
        /// The diarizer: `auto` (the catalog's diarizer entry), a catalog
        /// entry's id, or `off`. It is enabled when its files are on disk;
        /// nothing is ever downloaded by itself. A text, image or speech
        /// model's id is refused. Loaded by the first session that asks for
        /// speakers, unloaded with the speech model.
        model: Text = super::DEFAULT_DIARIZER_MODEL.to_owned(), NO_RULE, ALL, [Visible, Patchable];
        /// The threads the diarizer computes with, a pool of its own beside the
        /// speech model's, so its steps never wait for the speech model's work:
        /// `auto` is a third of the
        /// physical cores, at least 2 (3 on a 10-core machine), counted within the CPUs the server may run on.
        /// Read by the load, so a change holds from the next one; the load
        /// logs the count (`ignis.diarizer.load`).
        threads: Threads = None, NO_RULE, ALL, [Visible, Patchable];
        /// The ghost rule, in seconds of speech: a voice the diarizer hears is
        /// a speaker of its own once it has spoken this long, and until then
        /// its speech counts for the speaker it sounds most like, so a short
        /// mistaken voice splits no utterance. The session's first voice is a
        /// speaker at once. 0 makes every voice a speaker at its first frame.
        /// At 2 s a ghost voice of the synthetic conversations goes, and no
        /// real meeting's quiet participant is lost (spec speech/08, As built).
        min_speaker_seconds: Decimal = 2_000, decimal(0.0, 60.0), ALL, [Visible, Patchable];
        /// The live geometry's speaker cache, in 80 ms frames: what the
        /// diarizer keeps of every speaker heard so far.
        live_speaker_cache_frames: Frames = 264, between(8, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The live geometry's FIFO, in 80 ms frames: the latest frames each
        /// step reads beside its chunk.
        live_fifo_frames: Frames = 264, between(0, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The live geometry's chunk, in 80 ms frames: the frames one step
        /// scores. 25 is 2 s.
        live_chunk_frames: Frames = 25, between(1, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The live geometry's right context, in 80 ms frames: the look-ahead
        /// a step waits for. With the chunk, the latency: 2.32 s at 25 and 4.
        live_right_context_frames: Frames = 4, between(0, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The live geometry's update period, in 80 ms frames: the oldest
        /// frames an overflowing FIFO gives the speaker cache at once.
        live_update_period_frames: Frames = 222, between(1, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The file geometry's speaker cache, in 80 ms frames, for a session
        /// fed faster than real time (`session.ignis.diarization: "file"`).
        file_speaker_cache_frames: Frames = 264, between(8, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The file geometry's FIFO, in 80 ms frames.
        file_fifo_frames: Frames = 40, between(0, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The file geometry's chunk, in 80 ms frames: 340 is 27.2 s.
        file_chunk_frames: Frames = 340, between(1, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The file geometry's right context, in 80 ms frames: with the chunk,
        /// 30.4 s of latency at 340 and 40.
        file_right_context_frames: Frames = 40, between(0, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The file geometry's update period, in 80 ms frames.
        file_update_period_frames: Frames = 300, between(1, 1000), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The speaker cache's floor on a frame's probabilities when it scores
        /// the frames to keep. `auto` is the checkpoint's (0.25).
        prediction_score_threshold: AutoDecimal = None, decimal(0.001, 1.0), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// What the speaker cache adds to the scores of the frames the FIFO
        /// just gave it. `auto` is the checkpoint's (0.05).
        latest_frames_score_boost: AutoDecimal = None, decimal(0.0, 10.0), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The share of each speaker's room in the speaker cache whose best
        /// frames get the strong boost. `auto` is the checkpoint's (0.75).
        strong_boost_rate: AutoDecimal = None, decimal(0.0, 4.0), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The share of each speaker's room in the speaker cache whose best
        /// frames get the weak boost. `auto` is the checkpoint's (1.5).
        weak_boost_rate: AutoDecimal = None, decimal(0.0, 4.0), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The share of each speaker's room a speaker's positive scores must
        /// fill before its other frames stop counting. `auto` is the
        /// checkpoint's (0.5).
        min_positive_scores_rate: AutoDecimal = None, decimal(0.0, 1.0), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
        /// The silence frames the speaker cache keeps for every speaker.
        /// `auto` is the checkpoint's (1).
        speaker_cache_silence_frames_per_speaker: AutoCount = None, between(0, 125), NEMOTRON_DIARIZATION_ONLY, [Visible, Patchable];
    }
}

config_group! {
    /// Fetching a missing model.
    DownloadGroup = "download" {
        /// Fetch the model when no artifact is on disk — asked first when
        /// stdin is a terminal, downloaded straight away when it is not. Off
        /// keeps the placeholder template, and refuses POST
        /// /v1/models/download too: the server never uses the network.
        enabled: Bool = super::DEFAULT_MODEL_DOWNLOAD, NO_RULE, ALL, [Visible];
        /// Where a fetched model lands, and where one fetched earlier is
        /// found.
        path: Path = PathBuf::from(super::DEFAULT_MODEL_DOWNLOAD_PATH), NO_RULE, ALL, [Visible];
        /// Where a fetched model's files come from: Hugging Face, or a mirror
        /// answering its {repo}/resolve/{revision}/{file} route (an
        /// Artifactory or Nexus proxy, a static server laid out the same
        /// way). The pins still say whether the bytes are the published ones.
        endpoint: Text = super::DEFAULT_DOWNLOAD_ENDPOINT.to_owned(), NO_RULE, ALL, [Visible];
        /// Sent as `Authorization: Bearer` to the endpoint, never across a
        /// redirect to another host. Unset, HF_TOKEN is sent to huggingface.co
        /// and nothing to a mirror. Never shown by GET /v1/config or `config
        /// print`.
        token: Opt<Token> = None, NO_RULE, ALL;
        /// An operator catalog: a YAML or JSON file of models fetched beside
        /// the built-in ones, in the same format; an id the built-in catalog
        /// has is refused. A relative path is read from the directory of the
        /// config file that names it.
        catalog: Opt<Path> = None, NO_RULE, ALL, [Visible];
        /// Let POST /v1/models/download, the Server view's Download, fetch a
        /// catalog entry while the server runs, under the same API key as
        /// PATCH /v1/config; it needs `enabled` too. Off refuses it, and
        /// `model download` on the server's machine still fetches. Off
        /// refuses POST /v1/models/remove too, the Server view's Remove.
        from_ui: Bool = true, NO_RULE, ALL, [Visible];
        /// What a download must leave free on the volume of `path` beyond
        /// the bytes it still misses: the Server view's Download, `model
        /// download` and the start's own fetch are refused, before any byte
        /// is written, when the free space is below the two together.
        free_margin: Bytes = super::DEFAULT_DOWNLOAD_FREE_MARGIN, NO_RULE, ALL, [Visible];
    }
}

config_group! {
    /// Flash-Next's n-gram table.
    NgramGroup = "ngram" {
        /// Persist Flash-Next's n-gram hot rows between loads.
        persist: Bool = true, NO_RULE, ALL, [Visible, Patchable, ReloadRequired];
        /// Where the n-gram cache goes: `model` (beside the artifact), `auto`
        /// (Windows LOCALAPPDATA/ignis/cache/ngram, Linux
        /// XDG_CACHE_HOME/ignis/ngram or HOME/.cache/ignis/ngram), or a
        /// directory.
        persist_path: Location = CacheLocation::Model, NO_RULE, ALL, [Visible, Patchable, ReloadRequired];
        /// The n-gram rows held in RAM, every other row read from NVMe when a
        /// step needs it: a byte count, or `auto` — what the host plan leaves
        /// after its other lines and the margin, less 256 MiB for the
        /// load, the whole ~29 GB table when that fits, never below the
        /// default. Unset, 1 GiB.
        hot_bytes: Opt<HotBytes> = None, NO_RULE, FLASH_NEXT_ONLY, [Visible, Patchable, ReloadRequired];
    }
}

config_group! {
    /// KV-disk, the tier below KV-RAM: evicted sequences and retained prompt
    /// checkpoints kept as files and read back instead of prefilled again.
    KvDiskGroup = "kv_disk" {
        /// KV-disk's budget: a ceiling, cut at start to the volume's free
        /// space less 10 GiB. Unset, the model's own: 4G on Flash-Next, 0
        /// (off) on the 27B.
        bytes: Opt<Bytes> = None, NO_RULE, ALL, [Visible, Patchable, ReloadRequired, Scoped];
        /// Where KV-disk's files go: `model` (beside the artifact), `auto`
        /// (the per-user cache directory under kv-disk), or a directory. They
        /// go in an ignis-kv-disk/<pid>-<nonce> directory there, removed at
        /// shutdown.
        path: Location = CacheLocation::Model, NO_RULE, ALL, [Visible, Patchable, ReloadRequired];
    }
}

settings! {
    /// The HTTP server.
    server: ServerGroup,
    /// The model and its load shape.
    model: ModelGroup,
    /// Device memory.
    vram: VramGroup,
    /// Prompt reuse and KV-RAM.
    reuse: ReuseGroup,
    /// The model switch.
    switch: SwitchGroup,
    /// Speculative decoding.
    spec: SpecGroup,
    /// The vision tower.
    vision: VisionGroup,
    /// Media acquisition.
    media: MediaGroup,
    /// The image model's serving.
    image: ImageGroup,
    /// The speech model.
    speech: SpeechGroup,
    /// The diarizer.
    diarizer: DiarizerGroup,
    /// Model download.
    download: DownloadGroup,
    /// The n-gram table.
    ngram: NgramGroup,
    /// KV-disk.
    kv_disk: KvDiskGroup,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The struct's real field names, read from its derived `Debug` — which
    /// rustc writes from the struct definition, not from the macro's table —
    /// so this compares two independent lists.
    fn debug_field_names(value: &impl std::fmt::Debug) -> Vec<String> {
        format!("{value:#?}")
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix("    ")?;
                if rest.starts_with(' ') {
                    return None;
                }
                let (name, _) = rest.split_once(':')?;
                name.chars().all(|c| c.is_ascii_lowercase() || c == '_').then(|| name.to_owned())
            })
            .collect()
    }

    /// Spec config-v2/01, Testing: every group's table has exactly one entry
    /// per struct field and no orphan — the drift `IGNIS_PROMETHEUS` was.
    #[test]
    fn every_group_table_has_exactly_one_entry_per_struct_field() {
        let groups: Vec<(&str, Vec<String>)> = vec![
            (ServerGroup::NAME, debug_field_names(&ServerGroup::default())),
            (ModelGroup::NAME, debug_field_names(&ModelGroup::default())),
            (VramGroup::NAME, debug_field_names(&VramGroup::default())),
            (ReuseGroup::NAME, debug_field_names(&ReuseGroup::default())),
            (SwitchGroup::NAME, debug_field_names(&SwitchGroup::default())),
            (SpecGroup::NAME, debug_field_names(&SpecGroup::default())),
            (VisionGroup::NAME, debug_field_names(&VisionGroup::default())),
            (MediaGroup::NAME, debug_field_names(&MediaGroup::default())),
            (ImageGroup::NAME, debug_field_names(&ImageGroup::default())),
            (SpeechGroup::NAME, debug_field_names(&SpeechGroup::default())),
            (DiarizerGroup::NAME, debug_field_names(&DiarizerGroup::default())),
            (DownloadGroup::NAME, debug_field_names(&DownloadGroup::default())),
            (NgramGroup::NAME, debug_field_names(&NgramGroup::default())),
            (KvDiskGroup::NAME, debug_field_names(&KvDiskGroup::default())),
        ];
        assert_eq!(groups.len(), GROUPS.len(), "a group missing from one of the two lists");
        for ((name, struct_fields), (table_name, table)) in groups.iter().zip(GROUPS) {
            assert_eq!(name, table_name);
            let table_fields: Vec<String> = table.iter().map(|meta| meta.name.to_owned()).collect();
            assert!(!struct_fields.is_empty(), "{name}: no field read from Debug");
            assert_eq!(&table_fields, struct_fields, "{name}");
        }
        assert_eq!(
            debug_field_names(&Settings::default()),
            GROUPS.iter().map(|(name, _)| name.to_string()).collect::<Vec<_>>(),
            "every group in Settings, in order"
        );
    }

    /// The eleven groups spec config-v2/01 §Groups names, the image
    /// model's (spec image-model/05), the speech model's (spec speech/02) and
    /// the diarizer's (spec speech/08), no more.
    #[test]
    fn the_groups_are_the_specs() {
        let names: Vec<&str> = GROUPS.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            ["server", "model", "vram", "reuse", "switch", "spec", "vision", "media", "image", "speech", "diarizer", "download", "ngram", "kv_disk"]
        );
    }

    /// Spec speech/08, AC 14: every field that is a concept of Sortformer, the
    /// two geometries and the speaker cache's six constants, only Nemotron 3
    /// Diarization takes, under its plain key; the model, its threads and the
    /// ghost rule every diarizer does. The geometries default to the measured
    /// ones, the constants to the checkpoint's (`auto`), each override
    /// bounded, the threads to `auto`.
    #[test]
    fn the_sortformer_fields_are_nemotron_diarizations_own() {
        let shared = ["model", "threads", "min_speaker_seconds"];
        for meta in DiarizerGroup::FIELDS {
            let only = meta.applies == NEMOTRON_DIARIZATION_ONLY;
            assert_eq!(only, !shared.contains(&meta.name), "{}", meta.name);
            assert!(meta.visible && meta.patchable && !meta.reload_required && !meta.scoped, "{}", meta.name);
            assert_eq!(meta.home_family(), None, "{} keeps its plain key", meta.name);
            assert!(only == (meta.validator != NO_RULE) || meta.name == "min_speaker_seconds", "{} is bounded", meta.name);
        }
        let group = DiarizerGroup::default();
        let geometry = |speaker_cache, fifo, chunk, right_context, update_period| [speaker_cache, fifo, chunk, right_context, update_period];
        assert_eq!(
            geometry(group.live_speaker_cache_frames, group.live_fifo_frames, group.live_chunk_frames, group.live_right_context_frames, group.live_update_period_frames),
            [264, 264, 25, 4, 222]
        );
        assert_eq!(
            geometry(group.file_speaker_cache_frames, group.file_fifo_frames, group.file_chunk_frames, group.file_right_context_frames, group.file_update_period_frames),
            [264, 40, 340, 40, 300]
        );
        let constants = [
            group.prediction_score_threshold,
            group.latest_frames_score_boost,
            group.strong_boost_rate,
            group.weak_boost_rate,
            group.min_positive_scores_rate,
            group.speaker_cache_silence_frames_per_speaker,
        ];
        assert_eq!(constants, [None; 6], "auto: the checkpoint's own");
        assert_eq!((group.model.as_str(), group.threads, group.min_speaker_seconds), ("auto", None, 2_000));
        assert_eq!(field("diarizer", "weak_boost_rate").unwrap().validator.describe(), "0..=4");
    }

    /// No two fields share a spelling on any surface, family-scoped ones
    /// included, and none takes a name the CLI keeps for itself.
    #[test]
    fn every_flag_env_var_and_file_key_is_unique() {
        let mut flags = BTreeSet::new();
        let mut envs = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for meta in all_fields() {
            assert!(flags.insert(meta.flag()), "{}", meta.flag());
            assert!(envs.insert(meta.env()), "{}", meta.env());
            assert!(keys.insert(meta.file_key()), "{}", meta.file_key());
            if meta.scoped {
                for family in super::super::field::FAMILIES {
                    assert!(flags.insert(meta.scoped_flag(family)), "{}", meta.scoped_flag(family));
                    assert!(envs.insert(meta.scoped_env(family)), "{}", meta.scoped_env(family));
                    assert!(keys.insert(meta.scoped_file_key(family)), "{}", meta.scoped_file_key(family));
                }
            }
        }
        for reserved in ["--config", "--profile", "--help", "--version"] {
            assert!(!flags.contains(reserved), "{reserved}");
        }
        for reserved in [super::super::CONFIG_ENV, super::super::PROFILE_ENV] {
            assert!(!envs.contains(reserved), "{reserved}");
        }
    }

    /// Spec config-v2/02 AC 2 and 6: the key is never visible nor patchable;
    /// the bind address is visible and not patchable; nothing is
    /// `ReloadRequired` without being patchable (the attribute would mean
    /// nothing).
    #[test]
    fn the_secret_is_hidden_and_the_sockets_are_not_patchable() {
        let api_key = field("server", "api_key").unwrap();
        assert!(!api_key.visible && !api_key.patchable);
        // Spec model-download/02 AC 7: the download token is hidden as the
        // key is, and no download field changes while the server runs.
        let token = field("download", "token").unwrap();
        assert!(!token.visible && !token.patchable);
        for name in ["enabled", "path", "endpoint", "catalog"] {
            let meta = field("download", name).unwrap();
            assert!(meta.visible && !meta.patchable, "{name}");
        }
        let bind = field("server", "bind").unwrap();
        assert!(bind.visible && !bind.patchable);
        for name in ["metrics_bind", "ui"] {
            assert!(!field("server", name).unwrap().patchable, "{name}");
        }
        for meta in all_fields() {
            assert!(!meta.reload_required || meta.patchable, "{}", meta.file_key());
        }
    }

    /// A field declared with no attributes at all is closed on every side:
    /// spec config-v2/02's fail-closed default, pinned on a throwaway
    /// declaration so no real field has to be made hidden to prove it.
    #[test]
    #[allow(dead_code)]
    fn a_field_given_no_attribute_is_hidden_unpatchable_and_unscoped() {
        config_group! {
            /// A throwaway group.
            Throwaway = "throwaway" {
                /// Declared with no attribute list.
                plain: Bool = false, NO_RULE, ALL;
                /// Declared visible only.
                shown: Bool = false, NO_RULE, ALL, [Visible];
            }
        }
        let plain = &Throwaway::FIELDS[0];
        assert!(!plain.visible && !plain.patchable && !plain.reload_required && !plain.scoped);
        assert!(Throwaway::FIELDS[1].visible && !Throwaway::FIELDS[1].patchable);
        assert_eq!(plain.summary(), "Declared with no attribute list.");
    }

    /// Spec config-v2/01, Testing: a pinned regression per field with a real
    /// validator today.
    #[test]
    fn the_validators_that_existed_before_are_declared() {
        assert_eq!(field("model", "prefill_chunk").unwrap().validator, Validator::MultipleOf(128));
        assert_eq!(field("spec", "decode_lanes").unwrap().validator, between(1, 8));
        assert!(matches!(field("model", "kv_format").unwrap().validator, Validator::OneOf(names) if names.contains(&"bf16") && names.contains(&"hq-e8-2b")));
        assert_eq!(field("server", "request_timeout").unwrap().validator, between(0, 3600));
    }

    /// The `OneOf` lists are literal (a `const` cannot call `as_str`); each
    /// is checked here against the enum it spells, so a new variant fails a
    /// test instead of being refused by its own validator.
    #[test]
    fn every_one_of_list_names_exactly_what_its_kind_parses() {
        let names = |group, name| match field(group, name).unwrap().validator {
            Validator::OneOf(names) => names.to_vec(),
            other => panic!("{group}.{name}: {other:?}"),
        };
        assert_eq!(names("server", "system_message_policy"), SystemMessagePolicy::ALL.map(|p| p.as_str()).to_vec());
        assert_eq!(names("server", "developer_message_policy"), DeveloperMessagePolicy::ALL.map(|p| p.as_str()).to_vec());
        assert_eq!(
            names("model", "reasoning_effort"),
            crate::thinking::ReasoningEffort::ALL.map(|e| e.as_str()).to_vec()
        );
        assert_eq!(names("server", "expose"), super::super::Expose::NAMES.to_vec());
        assert_eq!(names("server", "timeout_reply"), crate::deadline::TimeoutReply::ALL.map(|r| r.as_str()).to_vec());
        for name in names("model", "kv_format") {
            assert!(KvFormat::parse(name).is_ok(), "{name}");
        }
        assert_eq!(names("model", "prefill_attention"), PrefillAttention::NAMES.to_vec());
        assert_eq!(names("model", "exception_weights"), ExceptionWeights::NAMES.to_vec());
        for name in names("spec", "backend") {
            assert!(SpecBackend::parse(name).is_ok(), "{name}");
        }
        for name in names("spec", "draft_head") {
            assert!(DraftHead::parse(name).is_ok(), "{name}");
        }
    }

    /// `help --fields` prints every field's description: none may be empty.
    #[test]
    fn every_field_has_a_description() {
        for meta in all_fields() {
            assert!(!meta.summary().is_empty(), "{}", meta.file_key());
        }
    }
}
