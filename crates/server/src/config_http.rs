//! `GET` and `PATCH /v1/config` (spec config-v2/02): the running
//! configuration, read and changed over the existing HTTP surface.
//!
//! What a client may see and change is each field's own declaration
//! (`config::schema`): `visible` decides `GET` — a whitelist, so a field
//! nobody marked is absent, never masked, and the API key is simply not
//! there — and `patchable` / `reload_required` decide `PATCH`. A patch is
//! applied one way, never two: when nothing it changes for the model running
//! needs a reload ([`needs_reload`]) it takes effect at once with no model
//! touched ([`crate::Server::apply_config`]); when even one such field
//! does, the **whole** patch goes through the model switch
//! #305 built (`crate::model_switch::reconfigure`) — same gate, drain,
//! teardown and load, same return to the old model when the load fails — so
//! the server is never left half old config and half new.
//!
//! A change is written to the config file in use only once it has taken
//! (at once for a live one, when the reloaded model serves for the other),
//! through the same merge `config patch` writes with, so the file never
//! describes a config nothing is running. With no file in use the change
//! still applies, and the answer says it was not written down.

use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;
use axum::extract::{Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

use crate::api::{error_response, ApiError};
use crate::config::field::{scope, FieldMeta, FAMILIES};
use crate::config::file::{self, Files};
use crate::config::kind::{FieldKind, KnownModels};
use crate::config::schema::{self, all_fields};
use crate::config::source::{Candidate, Layer, Resolution, Source};
use crate::config::{Config, ConfigError};
use crate::Server;
use ignis_core::compute::ModelFamily;

/// The running configuration and where a change to it is written down,
/// shared by the server and the model loader.
pub struct ConfigState {
    /// The config the running model was loaded with, every live change on
    /// top: fitted to its family, so the values shown are the ones in force.
    current: ArcSwap<Config>,
    /// The filesystem a change is written to.
    files: Arc<dyn Files + Send + Sync>,
    /// Held across read, change and publish, so two changes never
    /// interleave and lose one another.
    changing: Mutex<()>,
}

impl ConfigState {
    /// The state of a server running `config`, writing changes through
    /// `files` to the config file `config` was read from, if any.
    pub fn new(config: Config, files: Arc<dyn Files + Send + Sync>) -> Self {
        Self { current: ArcSwap::from_pointee(config), files, changing: Mutex::new(()) }
    }

    /// The running config, as of now.
    pub fn current(&self) -> Arc<Config> {
        self.current.load_full()
    }

    /// Make `config` the running one: a live change that applied, or a
    /// model a switch or reload made serve.
    pub fn publish(&self, config: Config) {
        self.current.store(Arc::new(config));
    }

    /// Hold changes off until the guard drops.
    pub(crate) fn changing(&self) -> std::sync::MutexGuard<'_, ()> {
        self.changing.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Write `patch` into the config file in use, if there is one (spec
    /// config-v2/02 §`PATCH` step 5), by [`file::rewrite`] — the merge
    /// `config patch` writes with.
    ///
    /// A field only one family takes goes under that family's own section,
    /// never the general one, its value and its removal alike
    /// ([`Layer::homed`], as the running configuration applies it): it can
    /// only ever apply to that family, so there it is both right and enough,
    /// and a start of the other model from the same file never sees it
    /// instead of being refused for it. Every other field goes where the
    /// patch put it.
    ///
    /// A removal (`null`) takes its key out, and the family section or group
    /// it empties (spec playground/05 §Decisions). A patch that leaves the
    /// document as it was — a removal of a key the file does not hold — does
    /// not rewrite it: the file keeps its comments and its layout.
    pub fn persist(&self, patch: &Layer) -> Persisted {
        let current = self.current();
        let Some(path) = current.file_path() else {
            return Persisted::No(NO_FILE.to_owned());
        };
        let patch = patch.homed();
        let removals = patch.removals().map(|(meta, family, _)| (meta, family)).collect();
        let written = file::changes_of(&patch)
            .and_then(|changes| file::rewrite(&*self.files, path, changes, removals, None))
            .and_then(|rewritten| {
                if !rewritten.changed {
                    return Ok(());
                }
                self.files
                    .write(path, &rewritten.text)
                    .map_err(|e| ConfigError(format!("{} cannot be written: {e}", path.display())))
            });
        match written {
            Ok(()) => {
                tracing::info!(name: "ignis.config.persisted", path = %path.display(), "the config file holds the change (written, or held already)");
                Persisted::Yes(path.display().to_string())
            }
            Err(error) => {
                tracing::warn!(name: "ignis.config.persist_failed", path = %path.display(), %error, "the config change applied but was not written down");
                Persisted::No(error.0)
            }
        }
    }
}

/// Why a change was not written down when no config file is in use.
const NO_FILE: &str = "no config file is in use: none was named by --config or IGNIS_CONFIG, and none was found \
                       where a start looks; `ignis-server config generate --out <path>` makes one";

/// Whether a change reached the config file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Persisted {
    /// Written, to this path.
    Yes(String),
    /// Not written, and why.
    No(String),
}

/// `config` as `GET /v1/config` shows it: the config-file document of its
/// values — every family section where a family's value differs — with every
/// field not marked `visible` removed, its key and all.
pub fn visible_document(config: &Config) -> Value {
    let mut document = file::effective_document(config);
    remove_invisible(&mut document, all_fields());
    document
}

/// Remove every field of `fields` not marked `visible` from `document`, in
/// its group and in each family section — gone, not masked, so a client
/// cannot tell a hidden field from one that does not exist.
pub fn remove_invisible<'a>(document: &mut Value, fields: impl Iterator<Item = &'a FieldMeta>) {
    for meta in fields.filter(|meta| !meta.visible) {
        let Some(group) = document.get_mut(meta.group).and_then(Value::as_object_mut) else {
            continue;
        };
        group.remove(meta.name);
        for family in FAMILIES {
            if let Some(section) = group.get_mut(scope(family)).and_then(Value::as_object_mut) {
                section.remove(meta.name);
            }
        }
    }
}

/// A config document: `<group>: { <field>: value, <family>: { <field>: value } }`,
/// the config file's own shape.
#[derive(ToSchema)]
#[schema(value_type = Object)]
pub(crate) struct ConfigDocument(#[allow(dead_code)] Value);

/// The `501` of a server with no config to show or change.
fn no_config() -> Response {
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "server_error",
        "config_unavailable",
        "this server was built without a running configuration (a test server); there is nothing to show or change",
    )
}

/// The header `GET /v1/config` says the model's state in (`serving`,
/// `switching` or `failed`, as `GET /v1/models` reports it): during a
/// reload the body is still the configuration running, and this says a
/// change is under way.
pub const STATUS_HEADER: &str = "ignis-model-status";

/// `GET /v1/config` — the running configuration.
#[utoipa::path(
    get,
    path = "/v1/config",
    tag = "config",
    operation_id = "get_config",
    summary = "The running configuration",
    description = "The configuration in force, in the config file's own shape: `<group>: { <field>: value }`, with a `qwen38` / `qwen38flashnext` section beside a group's fields wherever a model family's value differs. A group key is the shared value, resolved for no family; a family's section (the loaded model's too) holds its own values, so what the loaded model runs with is its section's, else the group's. Every live change is on top.\n\nOnly fields declared visible appear: a hidden field (the API key) is absent, not masked. `ignis-server help --fields` lists every field and whether it is shown.\n\nAnswers during a model switch too: until the switch lands it shows the configuration still running, and the `ignis-model-status` header says `switching` (as `GET /v1/models` reports it).\n\n`?format=yaml` answers the same configuration as the config file it amounts to: `application/yaml`, the text `ignis-server config print` writes for it, without the hidden fields. It is the configuration in force — live changes included — not the file on disk, whose comments it does not carry.",
    params(("format" = inline(Option<ConfigFormat>), Query, description = "The answer's format, JSON when absent; any other value is `400 config_invalid`.")),
    responses(
        (status = 200, description = "The running configuration.",
            content((ConfigDocument = "application/json"), (String = "application/yaml")),
            headers(("ignis-model-status" = String, description = "`serving`, `switching` or `failed`."))),
        (status = 400, description = "`format` is neither `json` nor `yaml` (`config_invalid`).", body = ApiError),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 501, description = "This server was built without a running configuration (`config_unavailable`).", body = ApiError),
    ),
)]
pub(crate) async fn get_config(State(server): State<Arc<Server>>, Query(query): Query<ConfigQuery>) -> Response {
    let Some(state) = &server.config else {
        return no_config();
    };
    let format = match query.format.as_deref().map(ConfigFormat::named) {
        None => ConfigFormat::Json,
        Some(Some(format)) => format,
        Some(None) => {
            let other = query.format.unwrap_or_default();
            return refused("config_invalid", format!("`format={other}`: the formats are json and yaml"));
        }
    };
    let document = visible_document(&state.current());
    let mut response = match format {
        ConfigFormat::Json => Json(document).into_response(),
        // Spec playground/05 §Decisions: the rendered file — `config
        // print`'s writer over the document the JSON answer is.
        ConfigFormat::Yaml => ([(header::CONTENT_TYPE, "application/yaml")], file::Format::Yaml.write(&document)).into_response(),
    };
    response.headers_mut().insert(STATUS_HEADER, HeaderValue::from_static(server.status().as_str()));
    response
}

/// `GET /v1/config`'s query, as given: a format it does not name is refused
/// by the handler, as a `config_invalid` naming it, rather than by the
/// extractor.
#[derive(Deserialize)]
pub(crate) struct ConfigQuery {
    /// A [`ConfigFormat`]'s name.
    format: Option<String>,
}

/// What `GET /v1/config` answers in.
#[derive(Serialize, ToSchema, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ConfigFormat {
    /// The configuration as a JSON document (the default).
    Json,
    /// The config file it amounts to, as `ignis-server config print` writes
    /// it (`application/yaml`).
    Yaml,
}

impl ConfigFormat {
    /// The format `?format=` names, spelled exactly as the document lists
    /// it: unlike `config print --format`, no `yml`, no other case.
    fn named(name: &str) -> Option<Self> {
        match name {
            "json" => Some(Self::Json),
            "yaml" => Some(Self::Yaml),
            _ => None,
        }
    }
}

/// One field's declaration as `GET /v1/config/fields` serves it. Documentation
/// only: the route builds each object with [`FieldMeta::declaration`], the
/// function `help --fields --format json` builds its objects with too.
#[derive(Serialize, ToSchema)]
#[allow(dead_code)]
pub(crate) struct FieldDeclaration {
    /// The group, as the file key spells it (`reuse`).
    group: String,
    /// The field's name within its group (`kv_host_pool_bytes`).
    name: String,
    /// `<group>.<name>`, as a config file nests it: what a `400` from `PATCH
    /// /v1/config` names.
    file_key: String,
    /// The command-line flag (`--reuse-kv-host-pool-bytes`).
    flag: String,
    /// The environment variable (`IGNIS_REUSE_KV_HOST_POOL_BYTES`).
    env: String,
    /// The value's kind (`bool`, `bytes`, `tokens|off`, …).
    kind: String,
    /// The field may be unset: its value is `null`, which a page shows as
    /// "auto".
    nullable: bool,
    /// The default, as `GET /v1/config` writes a value.
    #[schema(value_type = Object)]
    default: Value,
    /// The rule a value must meet: `null`, `{ "min", "max" }` (an open end is
    /// `null`), `{ "multiple_of" }` or `{ "one_of": [..] }`.
    #[schema(value_type = Option<Object>)]
    rule: Option<Value>,
    /// The rule as a sentence (`1..=8`, `nonzero multiple of 128`, `one of
    /// bf16, hq-e8-2b`); empty with no rule.
    rule_text: String,
    /// `null` for a field every family takes, else the scope names
    /// (`qwen38`, `qwen38flashnext`) of the families that do.
    families: Option<Vec<String>>,
    /// The field may be given per family, in a `qwen38` / `qwen38flashnext`
    /// section beside the group's own value.
    scoped: bool,
    /// `PATCH /v1/config` may change it; else it is set at start.
    patchable: bool,
    /// A change takes effect only through a reload of the loaded model.
    reload_required: bool,
    /// What the field means, on one line.
    description: String,
}

/// The envelope of `GET /v1/config/fields`, as `GET /v1/models`' is: a key
/// can be added beside `data` without breaking a client.
#[derive(Serialize, ToSchema)]
#[allow(dead_code)]
pub(crate) struct FieldList {
    /// Always `list`.
    object: &'static str,
    /// The visible fields, in declaration order.
    data: Vec<FieldDeclaration>,
}

/// The visible fields' declarations, in declaration order — the same
/// whitelist `GET /v1/config` shows, so a hidden field (the API key, the
/// download token) is absent here as there.
pub fn visible_declarations() -> Vec<Value> {
    all_fields().filter(|meta| meta.visible).map(|meta| Value::Object(meta.declaration())).collect()
}

/// `GET /v1/config/fields` — what each visible field is.
#[utoipa::path(
    get,
    path = "/v1/config/fields",
    tag = "config",
    operation_id = "get_config_fields",
    summary = "The configuration's fields",
    description = "One object per field `GET /v1/config` shows, in declaration order: the field's group and name, the key a config file and a `400` from `PATCH /v1/config` spell it by, its kind, its default, the rule a value must meet (structured in `rule`, as a sentence in `rule_text`), the model families that take it (`null` for all), whether it may be unset (`nullable`), given per family (`scoped`), changed while the server runs (`patchable`) and whether a change reloads the model (`reload_required`). The same declarations `ignis-server help --fields --format json` prints, minus the fields `GET /v1/config` hides.\n\nStatic: it answers whatever state the model is in — during a switch, after a failed one — and on a server with no running configuration.",
    responses(
        (status = 200, description = "The visible fields, in declaration order.", body = FieldList),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
    ),
)]
pub(crate) async fn get_config_fields() -> Json<Value> {
    Json(serde_json::json!({ "object": "list", "data": visible_declarations() }))
}

/// The source of a value, as `GET /v1/config/origins` names it: the
/// resolver's [`Source`], a live patch called what it is to a client.
#[derive(Serialize, ToSchema, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub(crate) enum OriginSource {
    /// The field's hardcoded default: no source sets it.
    Default,
    /// The hardware profile in use: a default, never an override.
    Profile,
    /// The config file.
    File,
    /// An environment variable.
    Env,
    /// The card, which decided a value no one named: the band it runs.
    Card,
    /// A command-line flag.
    Flag,
    /// A `PATCH /v1/config` since the server started.
    Live,
}

impl From<Source> for OriginSource {
    fn from(source: Source) -> Self {
        match source {
            Source::Patch => Self::Live,
            Source::Flag => Self::Flag,
            Source::Env => Self::Env,
            Source::File => Self::File,
            Source::Card => Self::Card,
            Source::Profile => Self::Profile,
            Source::Default => Self::Default,
        }
    }
}

/// Where one field's value in one scope comes from.
#[derive(Serialize, ToSchema)]
pub(crate) struct OriginRow {
    /// The field, as a config file keys it (`reuse.kv_host_pool_bytes`).
    key: String,
    /// `null` for the shared value (the group key), else the scope name of
    /// a family that takes the field (`qwen38`, `qwen38flashnext`).
    scope: Option<&'static str>,
    /// The source of the value that resolves in this scope.
    source: OriginSource,
    /// The value was set for this family itself: a family section of the
    /// config file or of the profile, a `--<family>-…` flag, an
    /// `IGNIS_<FAMILY>_…` env var, a live family value. Never on the shared
    /// value's row.
    scoped: bool,
    /// Where the value was set, as the startup log spells it: the flag, the
    /// env var, the file key and the file's path, the profile's name, or the
    /// card's name and compute capability. Absent for a default and for a live
    /// change.
    #[serde(skip_serializing_if = "Option::is_none")]
    spelling: Option<String>,
}

/// The envelope of `GET /v1/config/origins`, as `GET /v1/models`' is.
#[derive(Serialize, ToSchema)]
pub(crate) struct OriginList {
    /// Always `list`.
    object: &'static str,
    /// For each visible field in declaration order, the shared value's row,
    /// then one per family that takes the field.
    data: Vec<OriginRow>,
}

/// Every visible field's origin in each scope (spec playground/05
/// follow-up 2), read off the very resolutions [`visible_document`]'s values
/// come from ([`file::resolutions`]): the source the resolver recorded for
/// the winning value, so a value and its origin cannot disagree. A field
/// the resolver recorded nothing for is at its hardcoded default.
fn visible_origins(config: &Config) -> Vec<OriginRow> {
    let (shared, families) = file::resolutions(config);
    let profile = &config.basis.sources().profile_name;
    let row = |meta: &FieldMeta, family: Option<ModelFamily>, resolution: &Resolution| {
        let origin = resolution.origin(meta.group, meta.name);
        let source = origin.map_or(Source::Default, |origin| origin.source);
        // A profile's values arrive spelled by their key in the profile's
        // definition; the operator chose the profile, by name.
        let spelling = match (source, origin) {
            (Source::Profile, _) => Some(profile.clone()),
            (Source::Flag | Source::Env | Source::File | Source::Card, Some(origin)) => Some(origin.spelling.clone()),
            _ => None,
        };
        let scoped = origin.is_some_and(|origin| origin.scoped);
        OriginRow { key: meta.file_key(), scope: family.map(scope), source: source.into(), scoped, spelling }
    };
    let mut rows = Vec::new();
    for meta in all_fields().filter(|meta| meta.visible) {
        rows.push(row(meta, None, &shared));
        for (family, resolution) in families.iter().filter(|(family, _)| meta.applies.to(*family)) {
            rows.push(row(meta, Some(*family), resolution));
        }
    }
    rows
}

/// `GET /v1/config/origins` — where each running value comes from.
#[utoipa::path(
    get,
    path = "/v1/config/origins",
    tag = "config",
    operation_id = "get_config_origins",
    summary = "Where each configuration value comes from",
    description = "One row per field `GET /v1/config` shows and per scope: `scope: null` for the shared value (the group key), then one row per model family that takes the field (`qwen38`, `qwen38flashnext`). `source` is where the value in force in that scope comes from: `default` (the hardcoded default), `profile` (the hardware profile in use, a default and never an override), `card` (the card the start read, which decided a value no one named: the prompt attention's FP16 band on a card below compute capability 12.0), `file` (the config file), `env`, `flag`, or `live` (a `PATCH /v1/config` since the server started). `scoped` is true when the value was set for that family itself — a family section of the file or of the profile, a `--<family>-…` flag, an `IGNIS_<FAMILY>_…` env var, a live family value. `spelling` names where, as the startup log does: the flag, the env var, the file key and the file's path, the profile's name, or the card's name and compute capability; it is absent for a default and a live change.\n\nThe value each row describes is the one `GET /v1/config` shows in that scope: one resolution decides both. A family's own value from the file or a live change (`scoped`, `source` `file` or `live`) is the operator's: a patched group value does not reach that family, and a `null` in the family's section removes it. A family's value from a profile, a flag or an env var is not removed by a `null`: a patched group value replaces it while the server runs, and a flag or an env var takes it back at the next start (`shadowed_at_restart` says so).\n\nAnswers during a model switch, showing the configuration still running.",
    responses(
        (status = 200, description = "Each visible field's origin, shared value first, then each family's.", body = OriginList),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 501, description = "This server was built without a running configuration (`config_unavailable`).", body = ApiError),
    ),
)]
pub(crate) async fn get_config_origins(State(server): State<Arc<Server>>) -> Response {
    let Some(state) = &server.config else {
        return no_config();
    };
    Json(OriginList { object: "list", data: visible_origins(&state.current()) }).into_response()
}

/// How a patch was applied.
#[derive(Serialize, ToSchema)]
pub(crate) struct PatchApplied {
    /// `live` (applied at once, no model touched) or `reload` (the whole
    /// patch goes through a model reload, under way).
    applied: &'static str,
    /// Whether the change is written to the config file (spec config-v2/02
    /// §`PATCH` step 6). A reload's change is written once the reloaded model
    /// serves, so its answer is `false` with that reason.
    persisted: bool,
    /// The config file the change was, or will be, written to.
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    /// Why the change is not (yet) written down.
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    /// The fields of the patch a flag or an env var also sets: those outrank
    /// the config file at the next start, so the written value applies only
    /// until then — each named with the spelling that sets it, a family's
    /// own flag or env var included for a group key ([`shadowed_at_restart`]).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    shadowed_at_restart: Vec<String>,
    /// The reload under way: the id served when it began, and the id it
    /// loads (the same model, reloaded with the patched config).
    #[serde(skip_serializing_if = "Option::is_none")]
    switching: Option<SwitchingIds>,
    /// The running configuration after a live change, as `GET /v1/config`
    /// shows it.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Object>)]
    config: Option<Value>,
}

/// The two ends of a reload.
#[derive(Serialize, ToSchema)]
pub(crate) struct SwitchingIds {
    from: String,
    to: String,
}

/// A refused patch: nothing in it was applied.
fn refused(code: &str, message: impl Into<String>) -> Response {
    error_response(StatusCode::BAD_REQUEST, "invalid_request_error", code, message)
}

/// Read a `PATCH` body into a layer, refusing — whole — what a patch may not
/// carry: a profile (profiles are config-file content), a field that is not
/// `patchable`, a field or a value the config file would refuse. A `null`
/// removes the field's key ([`file::read_merge_patch`]), and is refused as
/// a value would be.
pub fn read_patch(body: &Value) -> Result<Layer, (&'static str, String)> {
    if let Some(key) = ["profile", "profiles"].into_iter().find(|key| body.get(key).is_some()) {
        return Err(("config_invalid", format!("`{key}` is config-file content, not a live change: edit the file and restart")));
    }
    if !body.is_object() {
        return Err(("config_invalid", "the body must be a config document: `{ <group>: { <field>: value } }`".to_owned()));
    }
    let mut patch = file::read_merge_patch(body, "the request body").map_err(|e| ("config_invalid", e.0))?;
    drop_draft_options_under_off(&mut patch).map_err(|e| ("config_invalid", e.0))?;
    if patch.is_empty() {
        return Err(("config_invalid", "the patch names no field".to_owned()));
    }
    let mut refused: Vec<String> = patch.named().filter(|(meta, _)| !meta.patchable).map(|(meta, _)| meta.file_key()).collect();
    refused.dedup();
    if !refused.is_empty() {
        return Err((
            "config_field_not_patchable",
            format!(
                "{} cannot be changed while the server runs (not patchable); nothing in the patch was applied",
                refused.iter().map(|key| format!("`{key}`")).collect::<Vec<_>>().join(", ")
            ),
        ));
    }
    Ok(patch)
}

/// A patch that sets `spec.backend` to `off` and names no draft option beside
/// it removes the ones the same scope holds — in the patches before it and in
/// the config file — as it would if it said `null` for each (GitHub #329 item
/// 15). Speculation off has nothing to configure, and the draft count an
/// operator turns it off from is whatever an earlier layer left (a profile
/// value never refuses it; a file or live one would, without this): turning it
/// off is one step whichever layer set the count. A count the patch names in
/// the same scope, value or `null`, is the operator's own word and is judged
/// as written — `off` beside a count stays a refusal. A flag's or an env
/// var's count is outside a patch's reach and still refuses.
fn drop_draft_options_under_off(patch: &mut Layer) -> Result<(), ConfigError> {
    let backend = schema::field("spec", "backend").expect("a declared field");
    let off: Vec<Option<ModelFamily>> = patch
        .entries()
        .filter(|(meta, _, candidate)| {
            meta.name == backend.name && meta.group == backend.group && matches!(<crate::config::kind::SpecBackend as FieldKind>::parse(candidate.raw.trim()), Ok(crate::config::SpecChoice::Off))
        })
        .map(|(_, family, _)| family)
        .collect();
    for family in off {
        for name in ["draft_tokens", "draft_head"] {
            let meta = schema::field("spec", name).expect("a declared field");
            if patch.named().any(|(named, scoped)| std::ptr::eq(named, meta) && scoped == family) {
                continue;
            }
            let spelling = family.map_or_else(|| meta.file_key(), |family| meta.scoped_file_key(family));
            patch.remove(meta, family, spelling)?;
        }
    }
    Ok(())
}

/// A patch naming `switch.known_models` adds to the models already known
/// (spec config-v2/02 AC 4: "`known-model` additions") rather than
/// replacing them: the entries `running` lists are kept, the patch's added,
/// an id both name taking the patch's path. Done before the patch is
/// applied, so the merged list is what applies and what is written down.
pub fn add_known_models(patch: &mut Layer, running: &Config) -> Result<(), ConfigError> {
    let meta = schema::field("switch", "known_models").expect("a declared field");
    let Some(candidate) = patch.get(meta, None).cloned() else {
        return Ok(());
    };
    let added = KnownModels::parse(candidate.raw.trim()).map_err(|reason| ConfigError(format!("`{}` {reason}", candidate.spelling)))?;
    let mut known = running.known_models.clone();
    known.extend(added);
    let raw = known.iter().map(|(id, path)| format!("{id}={}", path.display())).collect::<Vec<_>>().join(";");
    patch.set(meta, None, Candidate { raw, spelling: candidate.spelling })
}

/// The fields of `patch`, set or removed, that a flag or an env var of
/// `running`'s start also sets, each with the spelling that sets it: written
/// to the file, they are outranked again at the next start (spec
/// config-v2/01 AC 6), so the answer says so rather than promise a change
/// that will not survive it. After a removal the flag or env var is in force
/// at once (spec playground/05 §Decisions).
///
/// What outranks a key is what a start resolving it would read first, above
/// the file: for a family's key, that family's flag, the general flag, its
/// env var, the general env var. A group key is read by the shared value and
/// by every family that has a section for the field, so each of those is
/// asked, and a family's own flag or env var is named too (spec playground/05
/// AC 30) — while the server runs, the patched group value outranks it, and
/// at the next start it takes that family back. A spelling is named once per
/// key.
pub fn shadowed_at_restart(patch: &Layer, running: &Config) -> Vec<String> {
    let sources = running.basis.sources();
    let set_at_start = |meta: &FieldMeta, family: Option<ModelFamily>| {
        [&sources.flags, &sources.env]
            .into_iter()
            .find_map(|layer| family.and_then(|family| layer.get(meta, Some(family))).or_else(|| layer.get(meta, None)))
    };
    let mut shadowed = Vec::new();
    for (meta, family) in patch.named() {
        let key = family.map_or_else(|| meta.file_key(), |family| meta.scoped_file_key(family));
        let readers: Vec<Option<ModelFamily>> = match family {
            Some(_) => vec![family],
            None => std::iter::once(None).chain(FAMILIES.into_iter().filter(|family| meta.takes_scope(*family)).map(Some)).collect(),
        };
        let mut named: Vec<&str> = Vec::new();
        for by in readers.into_iter().filter_map(|reader| set_at_start(meta, reader)) {
            if !named.contains(&by.spelling.as_str()) {
                named.push(&by.spelling);
                shadowed.push(format!("{key} (set by {})", by.spelling));
            }
        }
    }
    shadowed
}

/// Whether going from `running` to `next` — the running config with a
/// patch applied, both fitted to the family running ([`Config::with_patch`])
/// — needs a model reload: a field that takes effect only through one
/// resolves differently (spec playground/05 §Decisions). Judged on what the
/// running model would run with, not on what the patch names: a value scoped
/// to the family not loaded, or a group value the running family overrides
/// in its own section, changes nothing running, so it is a live change —
/// applied, written down, and used when that family next loads.
pub fn needs_reload(running: &Config, next: &Config) -> bool {
    let (before, after) = (running.settings().render(), next.settings().render());
    let value = |groups: &serde_json::Map<String, Value>, meta: &FieldMeta| groups.get(meta.group).and_then(|group| group.get(meta.name)).cloned();
    // The host margin is read by the text models' host plans (Flash-Next's,
    // and the 27B's since ADR 0062); the image model carries it for a later
    // switch, and is not reloaded for it.
    let plans_host = running.basis.family().is_none_or(|family| family != ModelFamily::QwenImage21);
    all_fields()
        .filter(|meta| meta.reload_required && (plans_host || meta.name != "host_margin_bytes"))
        .any(|meta| match (meta.group, meta.name) {
            // ADR 0057: the table the load runs, not the setting's spelling:
            // `auto` and the factor it resolves to are the same load.
            ("model", "rope_scaling") => running.rope_scaling != next.rope_scaling,
            _ => value(&before, meta) != value(&after, meta),
        })
}

/// The checks a start makes on the thinking defaults, made again on the
/// loaded model for the ones a live change names (spec server/08, GitHub
/// #68): a default the template cannot honour, or a budget its tokenizer has
/// no forced close for, is refused rather than silently inert. Only for what
/// `patch` changes — a value the load already accepted is not judged again —
/// and the budget's check only on a model loaded from an artifact, as at
/// start: the placeholder has no tokenizer to close with. Not on the image
/// model, which thinks nothing: the next text model's load judges them
/// (spec image-model/05).
fn check_thinking(server: &Server, next: &Config, patch: &Layer) -> Result<(), String> {
    let named = |name: &str| patch.named().any(|(meta, _)| meta.group == "model" && meta.name == name);
    let active = server.active();
    if active.engine().is_none() {
        return Ok(());
    }
    if named("enable_thinking") || named("reasoning_effort") {
        let defaults = crate::thinking::ThinkingDefaults {
            enable_thinking: next.enable_thinking,
            reasoning_effort: next.reasoning_effort,
        };
        crate::thinking::validate_defaults(&defaults, &active.template.thinking_capabilities())?;
    }
    if named("thinking_budget") && active.source.is_some() {
        let close = crate::thinking::thinking_close(|text| {
            active.template.encode_literal(text).ok_or_else(|| "the loaded template has no tokenizer".to_owned())
        });
        crate::thinking::check_default_budget_close(next.thinking_budget, &close)?;
    }
    Ok(())
}

/// Spec image-model/05 §Prompt enhancement: an `image.prompt_enhancer` a
/// patch names must be a text model a rewrite can run on — the loaded one, or
/// one an explicit switch by id may name, the patch's own
/// `switch.known_models` among them — where the server knows its text models
/// ([`crate::images::enhance::unknown_enhancer`]). Only for what `patch`
/// changes: an enhancer the running configuration already names is not
/// judged again (the start warned of it).
fn check_enhancer(server: &Server, next: &Config, patch: &Layer) -> Result<(), String> {
    if !patch.named().any(|(meta, _)| meta.group == "image" && meta.name == "prompt_enhancer") {
        return Ok(());
    }
    match crate::images::enhance::unknown_enhancer(server, &next.settings().image.prompt_enhancer, &next.known_models) {
        Some(why) => Err(why),
        None => Ok(()),
    }
}

/// `speech.model` and `speech.compat_names` against the catalog this server
/// runs with, an operator's file among it, when the patch names either: what
/// the start checks where it reads the catalog (`speech::check_names`); the
/// config's own check has the built-in entries already. `diarizer.model` the
/// same way (`speech::check_diarizer_name`).
fn check_speech(server: &Server, next: &Config, patch: &Layer) -> Result<(), String> {
    let Some(catalog) = server.speech.catalog() else {
        return Ok(());
    };
    if patch.named().any(|(meta, _)| meta.group == "speech" && (meta.name == "model" || meta.name == "compat_names")) {
        crate::speech::check_names(catalog, &next.settings().speech)?;
    }
    // `diarizer.model` (spec speech/08): the same kind check, for the diarizer.
    if patch.named().any(|(meta, _)| meta.group == "diarizer" && meta.name == "model") {
        crate::speech::check_diarizer_name(catalog, &next.settings().diarizer)?;
    }
    Ok(())
}

/// `PATCH /v1/config` — change the running configuration.
#[utoipa::path(
    patch,
    path = "/v1/config",
    tag = "config",
    operation_id = "patch_config",
    summary = "Change the running configuration",
    description = "The body is a partial config document in the config file's own shape — only the fields being changed, a `qwen38` / `qwen38flashnext` section for a family's own value. Each value is checked as the config file's would be; a field the loaded model's family cannot take, or one not patchable (the API key, the bind addresses, the Playground), refuses the **whole** patch, nothing in it applied. `switch.known_models` adds to the models already known.\n\nA `null` removes the key (JSON merge patch): in a family's section, that family's own value, so the group's applies to it again; at a group key, the group's value, so the profile's or the hardcoded default applies — or a flag or an env var that sets the field, which outranks the file (named in `shadowed_at_restart`). The value in force after a removal is the one a start from the rewritten file resolves; the file loses the key, and a family section or group left empty with it. A removal of a key nothing sets changes nothing. Only a field may be `null`, not a whole group or section. A field only one family takes is set and removed in that family's section, whichever key names it — in the configuration running as in the file.\n\nApplied one way, never two. When nothing the patch changes for the loaded model needs a reload — its fields take effect live (the request timeout, the message policies, the thinking defaults, the model switch's own knobs), or the reload fields it names resolve as before for the loaded model's family (a value in the other family's section, a group value the loaded family overrides in its own) — it takes effect at once, no model touched: `200`, with the configuration now running; a value for the other family is used when that family next loads. When a reload field resolves differently for the loaded model, the whole patch — its other fields included — goes through a reload of the loaded model with the patched configuration: the same switch `POST /v1/models/switch` runs, `202` at once, every other `/v1` route answering `503 model_switching` until it lands, and `GET /v1/models` reporting it. A reload that fails leaves the previous model serving with the previous configuration.\n\nA change is written to the config file in use once it has taken — at once, or when the reloaded model serves — keeping every other key in the file. With no config file in use it still applies, and `persisted: false` says why. `shadowed_at_restart` names the fields a flag or an env var also sets — for a group key, a model family's own `--<family>-…` flag or `IGNIS_<FAMILY>_…` variable too: they outrank the file at the next start.",
    request_body(content = ConfigDocument, description = "The fields to change, nested as the config file nests them."),
    responses(
        (status = 200, description = "Applied live.", body = PatchApplied),
        (status = 202, description = "A reload with the patched configuration began.", body = PatchApplied),
        (status = 400, description = "A field is unknown, not patchable (`config_field_not_patchable`), or wrong for the loaded model (`config_invalid`): nothing was applied.", body = ApiError),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 409, description = "A model switch is already under way (`switch_in_progress`), or the loaded model was not loaded from an artifact and cannot be reloaded (`reload_unavailable`).", body = ApiError),
        (status = 501, description = "This server has no running configuration, or no model loader to reload with.", body = ApiError),
        (status = 503, description = "The loaded model's first traversal has not run yet, or a model switch is under way; or a reload was needed while the server is stopping (`server_stopping`, with `Retry-After`): none begins, since the process would end under it.", body = ApiError),
    ),
)]
pub(crate) async fn patch_config(State(server): State<Arc<Server>>, Json(body): Json<Value>) -> Response {
    let Some(state) = server.config.clone() else {
        return no_config();
    };
    let mut patch = match read_patch(&body) {
        Ok(patch) => patch,
        Err((code, message)) => return refused(code, message),
    };
    let guard = state.changing();
    let running = state.current();
    if let Err(error) = add_known_models(&mut patch, &running) {
        return refused("config_invalid", error.0);
    }
    let next = match running.with_patch(&patch) {
        Ok(next) => next,
        Err(error) => return refused("config_invalid", error.0),
    };
    if let Err(error) = check_enhancer(&server, &next, &patch) {
        return refused("config_invalid", error);
    }
    if let Err(error) = check_speech(&server, &next, &patch) {
        return refused("config_invalid", error);
    }
    let shadowed = shadowed_at_restart(&patch, &running);
    if !needs_reload(&running, &next) {
        if let Err(error) = check_thinking(&server, &next, &patch) {
            return refused("config_invalid", error);
        }
        server.apply_config(&next);
        state.publish(next);
        let persisted = state.persist(&patch);
        drop(guard);
        let (persisted, path, reason) = match persisted {
            Persisted::Yes(path) => (true, Some(path), None),
            Persisted::No(reason) => (false, None, Some(reason)),
        };
        tracing::info!(name: "ignis.config.patched", applied = "live", persisted, "the running configuration changed");
        let config = Some(visible_document(&state.current()));
        let applied =
            PatchApplied { applied: "live", persisted, path, reason, shadowed_at_restart: shadowed, switching: None, config };
        return (StatusCode::OK, Json(applied)).into_response();
    }
    let path = running.file_path().map(|path| path.display().to_string());
    match crate::model_switch::reconfigure(&server, next, patch) {
        Ok(started) => {
            drop(guard);
            tracing::info!(
                name: "ignis.config.patched",
                applied = "reload",
                model = %started.to,
                "the running configuration changes with a reload of the loaded model"
            );
            let reason = match &path {
                Some(_) => "written once the reloaded model serves; not if the reload fails".to_owned(),
                None => NO_FILE.to_owned(),
            };
            let applied = PatchApplied {
                applied: "reload",
                persisted: false,
                path,
                reason: Some(reason),
                shadowed_at_restart: shadowed,
                switching: Some(SwitchingIds { from: started.from, to: started.to }),
                config: None,
            };
            (StatusCode::ACCEPTED, Json(applied)).into_response()
        }
        Err(refusal) => crate::api::switch_refused(refusal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::field::{Applicability, Attr, Validator};
    use crate::config::file::testing::MemFiles;
    use crate::config::kind::*;
    use crate::config::source::Resolver;
    use ignis_core::compute::ModelFamily;
    use serde_json::{json, Map};

    fn config(flags: &[&str]) -> Config {
        let args: Vec<String> = flags.iter().map(|s| s.to_string()).collect();
        match crate::config::resolve(&args, |_| None).unwrap() {
            crate::config::ConfigOutcome::Config(config) => config,
            other => panic!("{other:?}"),
        }
    }

    fn on_file(files: &MemFiles, flags: &[&str], env: &'static [(&'static str, &'static str)]) -> Config {
        let args: Vec<String> = flags.iter().map(|s| s.to_string()).collect();
        let env = move |key: &str| env.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string());
        match crate::config::resolve_with(&args, env, files).unwrap() {
            crate::config::ConfigOutcome::Config(config) => config,
            other => panic!("{other:?}"),
        }
    }

    /// Spec config-v2/02, Testing: the key started with is absent from the
    /// document — not `null`, not masked — and the bind address is there.
    #[test]
    fn the_api_key_is_absent_and_the_bind_address_present() {
        let config = config(&["--server-api-key", "sk-secret", "--server-bind", "127.0.0.1:7777"]);
        let document = visible_document(&config);
        assert!(document["server"].get("api_key").is_none(), "{document}");
        assert!(!document.to_string().contains("sk-secret"));
        assert_eq!(document["server"]["bind"], "127.0.0.1:7777");
        for meta in all_fields() {
            // A field only one family takes lives in that family's own
            // section, never the general one (GitHub #311's family-section
            // fix, also `effective_document`'s).
            let present = match meta.home_family() {
                Some(family) => document[meta.group].get(scope(family)).and_then(|s| s.get(meta.name)).is_some(),
                None => document[meta.group].get(meta.name).is_some(),
            };
            assert_eq!(present, meta.visible, "{}", meta.file_key());
        }
    }

    /// Spec model-download/02 AC 7: the download token is absent as the key
    /// is, the endpoint beside it shown; and no download field can be
    /// patched.
    #[test]
    fn the_download_token_is_absent_and_no_download_field_is_patchable() {
        let config = config(&["--download-token", "hf_secret", "--download-endpoint", "https://mirror.example.com"]);
        let document = visible_document(&config);
        assert!(document["download"].get("token").is_none(), "{document}");
        assert!(!document.to_string().contains("hf_secret"));
        assert_eq!(document["download"]["endpoint"], "https://mirror.example.com");
        let (code, message) = read_patch(&json!({ "download": { "token": "t", "endpoint": "https://x.example.com", "catalog": "c.yaml" } })).unwrap_err();
        assert_eq!(code, "config_field_not_patchable");
        assert!(["`download.token`", "`download.endpoint`", "`download.catalog`"].iter().all(|key| message.contains(key)), "{message}");
    }

    /// Spec config-v2/02, Testing: a field declared with no `visible` is
    /// left out — pinned on a throwaway declaration, so no real field has to
    /// be hidden to prove the whitelist is closed by default.
    #[test]
    #[allow(dead_code)]
    fn a_field_declared_without_visible_is_removed() {
        crate::config::schema::config_group! {
            /// A throwaway group.
            Throwaway = "throwaway" {
                /// Declared with no attribute list.
                plain: Bool = false, Validator::None, Applicability::AllFamilies;
                /// Declared visible.
                shown: Bool = false, Validator::None, Applicability::AllFamilies, [Visible];
            }
        }
        let mut document = json!({ "throwaway": { "plain": true, "shown": true, "qwen38": { "plain": true, "shown": true } } });
        remove_invisible(&mut document, Throwaway::FIELDS.iter());
        assert_eq!(document, json!({ "throwaway": { "shown": true, "qwen38": { "shown": true } } }));
    }

    /// GitHub #329 item 15: turning speculation off by `PATCH` is one step
    /// whichever layer set the draft count — the config file, or a patch
    /// before it. The count a patch inherits is dropped with the backend (and
    /// from the file, so a restart resolves the same); one the same patch
    /// names beside `off` is still refused, and so is one a flag sets.
    #[test]
    fn a_patch_turning_speculation_off_drops_the_draft_count_it_does_not_name() {
        let files = MemFiles::with(&[(
            "c.yaml",
            "spec:
  qwen38flashnext:
    backend: mtp
    draft_tokens: 3
",
        )]);
        let running = on_file(&files, &["--config", "c.yaml"], &[]).for_family(ModelFamily::FlashNext).unwrap();
        assert!(running.speculation.is_some());
        let off = json!({ "spec": { "qwen38flashnext": { "backend": "off" } } });

        // Set by the file.
        let patch = read_patch(&off).unwrap();
        let next = running.with_patch(&patch).unwrap();
        assert!(next.speculation.is_none() && next.speculation_off);
        assert!(patch.removals().any(|(meta, family, _)| meta.name == "draft_tokens" && family == Some(ModelFamily::FlashNext)));

        // Set by an earlier patch, on a start with no file value.
        let bare = config(&[]).for_family(ModelFamily::FlashNext).unwrap();
        let on = read_patch(&json!({ "spec": { "qwen38flashnext": { "backend": "mtp", "draft_tokens": 3 } } })).unwrap();
        let live = bare.with_patch(&on).unwrap();
        assert!(live.speculation.is_some());
        assert!(live.with_patch(&read_patch(&off).unwrap()).unwrap().speculation.is_none());

        // Named beside `off` in the same patch: still a refusal.
        let both = json!({ "spec": { "qwen38flashnext": { "backend": "off", "draft_tokens": 3 } } });
        let err = running.with_patch(&read_patch(&both).unwrap()).unwrap_err().0;
        assert!(err.contains("draft_tokens") && err.contains("off"), "{err}");

        // A flag's count is not the patch's to drop.
        let flagged = config(&["--spec-backend", "mtp", "--spec-draft-tokens", "3"]).for_family(ModelFamily::FlashNext).unwrap();
        let err = flagged.with_patch(&read_patch(&off).unwrap()).unwrap_err().0;
        assert!(err.contains("--spec-draft-tokens 3") && err.contains("off"), "{err}");
        // At the group key, the group's count goes with it.
        let group_on = read_patch(&json!({ "spec": { "backend": "mtp", "draft_tokens": 3 } })).unwrap();
        let group_off = read_patch(&json!({ "spec": { "backend": "off" } })).unwrap();
        assert!(bare.with_patch(&group_on).unwrap().with_patch(&group_off).unwrap().speculation.is_none());
    }

    #[test]
    fn a_patch_refuses_profiles_unknown_fields_and_anything_not_patchable_whole() {
        assert_eq!(read_patch(&json!({ "profile": "x" })).unwrap_err().0, "config_invalid");
        assert!(read_patch(&json!({ "server": { "bnd": "x" } })).unwrap_err().1.contains("server.bnd"));
        assert!(read_patch(&json!({})).unwrap_err().1.contains("names no field"));
        let (code, message) =
            read_patch(&json!({ "server": { "request_timeout": 60, "api_key": "k", "bind": "x" } })).unwrap_err();
        assert_eq!(code, "config_field_not_patchable");
        assert!(message.contains("`server.api_key`") && message.contains("`server.bind`") && !message.contains("request_timeout"), "{message}");
    }

    /// Spec playground/05 AC 21: a `null` is held to what a value there
    /// would be — a field not patchable, or a family section with no place
    /// for it, refuses the patch — and a body of only removals is a patch.
    #[test]
    fn a_removal_is_refused_where_a_value_would_be() {
        let (code, message) = read_patch(&json!({ "server": { "request_timeout": null, "api_key": null } })).unwrap_err();
        assert_eq!(code, "config_field_not_patchable");
        assert!(message.contains("`server.api_key`") && !message.contains("request_timeout"), "{message}");
        let (code, message) = read_patch(&json!({ "spec": { "qwen38": { "draft_rows": null } } })).unwrap_err();
        assert_eq!(code, "config_invalid");
        assert!(message.contains("spec.qwen38.draft_rows") && message.contains("does not take"), "{message}");
        let removals = read_patch(&json!({ "reuse": { "prompt": null } })).unwrap();
        assert_eq!(removals.removals().count(), 1);
    }

    /// Spec playground/05 AC 19-20: a patch reloads only when a field that
    /// needs a reload resolves differently for the family running — not for
    /// a value scoped to the other family, nor for a group value the running
    /// family overrides in its own section.
    #[test]
    fn a_patch_reloads_only_for_what_the_running_family_runs_with() {
        let running = config(&["--profile", "none"]).for_family(ModelFamily::Qwen38_27b).unwrap();
        let reloads = |running: &Config, body: Value| needs_reload(running, &running.with_patch(&read_patch(&body).unwrap()).unwrap());
        assert!(!reloads(&running, json!({ "server": { "request_timeout": 60 }, "switch": { "drain_timeout": 0 } })), "live fields");
        assert!(!reloads(&running, json!({ "reuse": { "qwen38flashnext": { "kv_host_pool_bytes": "1G" } } })), "the other family's value");
        assert!(!reloads(&running, json!({ "reuse": { "kv_host_pool_bytes": "2G" } })), "the value already running");
        assert!(reloads(&running, json!({ "reuse": { "qwen38": { "kv_host_pool_bytes": "1G" } } })), "the running family's own");
        assert!(reloads(&running, json!({ "reuse": { "kv_host_pool_bytes": "1G" } })), "a group value the running family reads");
        assert!(reloads(&running, json!({ "server": { "request_timeout": 60 }, "reuse": { "qwen38flashnext": { "prompt": false }, "prompt": false } })));
        let own = running.with_patch(&read_patch(&json!({ "reuse": { "qwen38": { "kv_host_pool_bytes": "4G" } } })).unwrap()).unwrap();
        assert!(!reloads(&own, json!({ "reuse": { "kv_host_pool_bytes": "1G" } })), "a group value the running family overrides");
    }

    /// ADR 0057: the reload check keys on the table the load runs, not on
    /// the setting's spelling: `auto` and the factor it resolves to at the
    /// load's context are the same load, for either model. A change of the
    /// table still reloads.
    #[test]
    fn a_rope_scaling_change_reloads_only_when_the_table_the_load_runs_changes() {
        let reloads = |running: &Config, body: Value| needs_reload(running, &running.with_patch(&read_patch(&body).unwrap()).unwrap());
        let rope = |value: &str| json!({ "model": { "rope_scaling": value } });
        let at = |context: &str, family| {
            config(&["--profile", "none", "--model-max-context", context, "--model-kv-format", "hq-e8-2b"]).for_family(family).unwrap()
        };
        let million = at("1048576", ModelFamily::Qwen38_27b);
        assert!(!reloads(&million, rope("yarn:4")), "auto is yarn:4 at 1,048,576");
        assert!(reloads(&million, rope("yarn:2")));
        assert!(reloads(&million, rope("none")));
        let pinned = million.with_patch(&read_patch(&rope("yarn:4")).unwrap()).unwrap();
        assert!(!reloads(&pinned, rope("auto")), "and back");
        let three = at("786432", ModelFamily::Qwen38_27b);
        assert!(!reloads(&three, rope("yarn:3")), "auto is yarn:3 at 786,432");
        assert!(reloads(&three, rope("yarn:4")));
        // Under `auto` a context change moves the table, and reloads.
        assert!(reloads(&three, json!({ "model": { "max_context": 1048576 } })));
        let flash_next = at("1048576", ModelFamily::FlashNext);
        assert!(!reloads(&flash_next, rope("yarn:4")), "auto is yarn:4 at 1,048,576 for Flash-Next too");
        assert!(reloads(&flash_next, rope("yarn:2")));
        assert!(reloads(&flash_next, rope("none")));
    }

    /// ADR 0057, ADR 0059: patching Flash-Next's own context while it runs
    /// re-resolves its `auto` table against the new context, and reloads; the
    /// 27B's context, patched meanwhile, moves nothing Flash-Next runs.
    #[test]
    fn a_patch_of_flash_nexts_context_re_resolves_its_rope_table_and_reloads() {
        let running = config(&["--profile", "none", "--qwen38flashnext-model-max-context", "589824"])
            .for_family(ModelFamily::FlashNext)
            .unwrap();
        assert_eq!(running.rope_scaling.factor(), 2.25);
        let patch = |body: Value| running.with_patch(&read_patch(&body).unwrap()).unwrap();
        let wider = patch(json!({ "model": { "qwen38flashnext": { "max_context": 1048576 } } }));
        assert_eq!((wider.max_context, wider.rope_scaling.factor()), (1_048_576, 4.0));
        assert!(needs_reload(&running, &wider));
        let trained = patch(json!({ "model": { "qwen38flashnext": { "max_context": 262144 } } }));
        assert_eq!(trained.rope_scaling, ignis_core::RopeScaling::NONE, "inside the trained position envelope");
        assert!(needs_reload(&running, &trained));
        let other = patch(json!({ "model": { "qwen38": { "max_context": 1048576 } } }));
        assert_eq!(other.rope_scaling, running.rope_scaling);
        assert!(!needs_reload(&running, &other), "the 27B's context is not Flash-Next's");
    }

    /// Spec playground/05: a `null` takes the patched key away, so
    /// `model.rope_scaling` is the default `auto` again and the table the
    /// load runs follows the context once more.
    #[test]
    fn a_null_rope_scaling_returns_to_auto_and_the_table_follows_the_context() {
        let running = config(&["--profile", "none", "--model-max-context", "1048576", "--model-kv-format", "hq-e8-2b"])
            .for_family(ModelFamily::Qwen38_27b)
            .unwrap();
        let pinned = running.with_patch(&read_patch(&json!({ "model": { "rope_scaling": "yarn:2" } })).unwrap()).unwrap();
        assert_eq!(pinned.settings().model.rope_scaling.to_string(), "yarn:2");
        assert_eq!(pinned.rope_scaling.factor(), 2.0);
        let removal = read_patch(&json!({ "model": { "rope_scaling": null } })).unwrap();
        let back = pinned.with_patch(&removal).unwrap();
        assert!(back.settings().model.rope_scaling.is_auto());
        assert_eq!(back.rope_scaling.factor(), 4.0, "auto at 1,048,576");
        assert!(needs_reload(&pinned, &back), "the table changed: 2 -> 4");

        // From a config file too: the key leaves the file and `auto` runs.
        let files = MemFiles::with(&[("c.yaml", "model:\n  rope_scaling: none\n  max_context: 786432\n")]);
        let on_disk = on_file(&files, &["--profile", "none", "--config", "c.yaml"], &[]).for_family(ModelFamily::Qwen38_27b).unwrap();
        assert_eq!(on_disk.rope_scaling, ignis_core::RopeScaling::NONE);
        let back = on_disk.with_patch(&removal).unwrap();
        assert!(back.settings().model.rope_scaling.is_auto());
        assert_eq!(back.rope_scaling.factor(), 3.0, "auto at 786,432");
    }

    /// The host margin is read by the text models' host plans -- Flash-Next's
    /// and, since ADR 0062 put its token embedding in pinned host memory, the
    /// 27B's: patching it reloads the family it names, and leaves the image
    /// model, which plans no host memory, running.
    #[test]
    fn the_host_margin_reloads_only_the_families_that_plan_host_memory() {
        let patch = |running: &Config, body: Value| needs_reload(running, &running.with_patch(&read_patch(&body).unwrap()).unwrap());
        let general = json!({ "model": { "host_margin_bytes": "8G" } });
        let flash_next_scoped = json!({ "model": { "qwen38flashnext": { "host_margin_bytes": "8G" } } });
        let qwen_scoped = json!({ "model": { "qwen38": { "host_margin_bytes": "8G" } } });
        let flash_next = config(&["--profile", "none"]).for_family(ModelFamily::FlashNext).unwrap();
        assert!(patch(&flash_next, general.clone()), "Flash-Next reads the general margin");
        assert!(patch(&flash_next, flash_next_scoped.clone()), "and its own");
        assert!(!patch(&flash_next, qwen_scoped.clone()), "not the 27B's");
        let qwen = config(&["--profile", "none"]).for_family(ModelFamily::Qwen38_27b).unwrap();
        assert!(patch(&qwen, general.clone()), "the 27B reads the general margin");
        assert!(patch(&qwen, qwen_scoped.clone()), "and its own");
        assert!(!patch(&qwen, flash_next_scoped.clone()), "not Flash-Next's");
        let image = config(&["--profile", "none"]).for_family(ModelFamily::QwenImage21).unwrap();
        assert!(!patch(&image, general), "the image model never reads the margin");
        assert!(!patch(&image, flash_next_scoped), "the image model: another family's key");
    }

    #[test]
    fn a_change_is_written_into_the_file_in_use_and_not_without_one_or_on_a_read_only_volume() {
        let files = Arc::new(MemFiles::with(&[("c.yaml", "server:\n  request_timeout: 30\nprofiles:\n  p:\n    vram:\n      headroom_bytes: 2G\n")]));
        let state = ConfigState::new(on_file(&files, &["--config", "c.yaml"], &[]), files.clone());
        let patch = read_patch(&json!({ "server": { "request_timeout": 90 } })).unwrap();
        assert_eq!(state.persist(&patch), Persisted::Yes("c.yaml".into()));
        let written = file::Format::Yaml.read(&files.files.lock().unwrap()[std::path::Path::new("c.yaml")]).unwrap();
        assert_eq!(written["server"]["request_timeout"], 90);
        assert_eq!(written["profiles"]["p"]["vram"]["headroom_bytes"], "2G", "the rest of the file is kept");

        let state = ConfigState::new(self::config(&[]), files.clone());
        let Persisted::No(reason) = state.persist(&patch) else { panic!("no file, nothing written") };
        assert!(reason.contains("no config file"), "{reason}");

        let read_only = Arc::new(MemFiles { read_only: true, ..MemFiles::with(&[("c.yaml", "server: {}\n")]) });
        let state = ConfigState::new(on_file(&read_only, &["--config", "c.yaml"], &[]), read_only.clone());
        let Persisted::No(reason) = state.persist(&patch) else { panic!("a write that fails is not written") };
        assert!(reason.contains("c.yaml") && reason.contains("cannot be written"), "{reason}");
    }

    /// Spec config-v2/02 AC 4: a patch of the known models adds to them.
    #[test]
    fn a_known_models_patch_adds_to_the_list_and_takes_the_patchs_path_for_a_shared_id() {
        let running = config(&["--switch-known-models", "a=F:/a.ninfer", "--switch-known-models", "b=F:/b.ninfer"]);
        let mut patch = read_patch(&json!({ "switch": { "known_models": { "b": "F:/b2.ninfer", "c": "F:/c.ninfer" } } })).unwrap();
        add_known_models(&mut patch, &running).unwrap();
        let next = running.with_patch(&patch).unwrap();
        let known: Vec<(&str, String)> = next.known_models.iter().map(|(id, path)| (id.as_str(), path.display().to_string())).collect();
        assert_eq!(known, [("a", "F:/a.ninfer".to_owned()), ("b", "F:/b2.ninfer".to_owned()), ("c", "F:/c.ninfer".to_owned())]);
    }

    /// A patched field a flag or an env var also sets is outranked again at
    /// the next start: the answer names it, with the spelling that sets it.
    #[test]
    fn a_patched_field_a_flag_or_env_var_sets_is_named_as_shadowed() {
        let files = MemFiles::with(&[("c.yaml", "server: {}\n")]);
        let running = on_file(&files, &["--config", "c.yaml", "--server-request-timeout", "40"], &[("IGNIS_MODEL_THINKING_BUDGET", "off")]);
        let patch = read_patch(&json!({ "server": { "request_timeout": 60, "system_message_policy": "strict" }, "model": { "thinking_budget": 1024 } })).unwrap();
        assert_eq!(
            shadowed_at_restart(&patch, &running),
            ["model.thinking_budget (set by IGNIS_MODEL_THINKING_BUDGET)", "server.request_timeout (set by --server-request-timeout)"]
        );
    }

    /// Spec playground/05 AC 30: a group key written to the file is
    /// outranked at the next start, for a family, by that family's own flag
    /// or env var — so each is named, once, beside a general one; a family's
    /// key only by its own family's and the general ones.
    #[test]
    fn a_family_flag_or_env_var_that_outranks_a_patched_group_value_is_named() {
        let files = MemFiles::with(&[("c.yaml", "server: {}\n")]);
        let env = &[("IGNIS_QWEN38FLASHNEXT_REUSE_KV_HOST_POOL_BYTES", "1G")];
        let running = on_file(&files, &["--config", "c.yaml", "--qwen38-reuse-kv-host-pool-bytes", "5G"], env);
        let shadowed = |body: Value| shadowed_at_restart(&read_patch(&body).unwrap(), &running);
        assert_eq!(
            shadowed(json!({ "reuse": { "kv_host_pool_bytes": "3G" } })),
            [
                "reuse.kv_host_pool_bytes (set by --qwen38-reuse-kv-host-pool-bytes)",
                "reuse.kv_host_pool_bytes (set by IGNIS_QWEN38FLASHNEXT_REUSE_KV_HOST_POOL_BYTES)"
            ]
        );
        assert_eq!(
            shadowed(json!({ "reuse": { "qwen38flashnext": { "kv_host_pool_bytes": null } } })),
            ["reuse.qwen38flashnext.kv_host_pool_bytes (set by IGNIS_QWEN38FLASHNEXT_REUSE_KV_HOST_POOL_BYTES)"],
            "the other family's flag is not this key's"
        );
        let general = on_file(&files, &["--config", "c.yaml", "--reuse-kv-host-pool-bytes", "6G", "--qwen38-reuse-kv-host-pool-bytes", "5G"], &[]);
        assert_eq!(
            shadowed_at_restart(&read_patch(&json!({ "reuse": { "kv_host_pool_bytes": "3G" } })).unwrap(), &general),
            ["reuse.kv_host_pool_bytes (set by --reuse-kv-host-pool-bytes)", "reuse.kv_host_pool_bytes (set by --qwen38-reuse-kv-host-pool-bytes)"],
            "the general flag once, for the shared value and every family without its own"
        );
        assert!(shadowed(json!({ "reuse": { "prompt": false } })).is_empty(), "nothing sets it");
    }
}
