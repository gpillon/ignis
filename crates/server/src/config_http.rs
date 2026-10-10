//! `GET` and `PATCH /v1/config` (spec config-v2/02): the running
//! configuration, read and changed over the existing HTTP surface.
//!
//! What a client may see and change is each field's own declaration
//! (`config::schema`): `visible` decides `GET` — a whitelist, so a field
//! nobody marked is absent, never masked, and the API key is simply not
//! there — and `patchable` / `reload_required` decide `PATCH`. A patch is
//! applied one way, never two: when no field in it needs a reload it takes
//! effect at once with no model touched ([`crate::Server::apply_config`]);
//! when even one does, the **whole** patch goes through the model switch
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
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use serde_json::Value;
use utoipa::ToSchema;

use crate::api::{error_response, ApiError};
use crate::config::field::{scope, FieldMeta, FAMILIES};
use crate::config::file::{self, Files};
use crate::config::kind::{FieldKind, KnownModels};
use crate::config::schema::{self, all_fields};
use crate::config::source::{Candidate, Layer};
use crate::config::{Config, ConfigError};
use crate::Server;

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
    pub fn persist(&self, patch: &Layer) -> Persisted {
        let current = self.current();
        let Some(path) = current.file_path() else {
            return Persisted::No(NO_FILE.to_owned());
        };
        let written = file::changes_of(patch)
            .and_then(|changes| file::rewrite(&*self.files, path, changes, None))
            .and_then(|text| {
                self.files
                    .write(path, &text)
                    .map_err(|e| ConfigError(format!("{} cannot be written: {e}", path.display())))
            });
        match written {
            Ok(()) => {
                tracing::info!(name: "ignis.config.persisted", path = %path.display(), "the config change is written to the config file");
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
    description = "The configuration in force, in the config file's own shape: `<group>: { <field>: value }`, with a `qwen38` / `qwen38flashnext` section beside a group's fields wherever a model family's value differs. The values are the ones the loaded model runs with — its family's own where one is set — with every live change on top.\n\nOnly fields declared visible appear: a hidden field (the API key) is absent, not masked. `ignis-server help --fields` lists every field and whether it is shown.\n\nAnswers during a model switch too: until the switch lands it shows the configuration still running, and the `ignis-model-status` header says `switching` (as `GET /v1/models` reports it).",
    responses(
        (status = 200, description = "The running configuration.", body = ConfigDocument,
            headers(("ignis-model-status" = String, description = "`serving`, `switching` or `failed`."))),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 501, description = "This server was built without a running configuration (`config_unavailable`).", body = ApiError),
    ),
)]
pub(crate) async fn get_config(State(server): State<Arc<Server>>) -> Response {
    let Some(state) = &server.config else {
        return no_config();
    };
    let mut response = Json(visible_document(&state.current())).into_response();
    response.headers_mut().insert(STATUS_HEADER, HeaderValue::from_static(server.status().as_str()));
    response
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
    /// until then — each named with the spelling that sets it.
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
/// `patchable`, a field or a value the config file would refuse.
pub fn read_patch(body: &Value) -> Result<Layer, (&'static str, String)> {
    if let Some(key) = ["profile", "profiles"].into_iter().find(|key| body.get(key).is_some()) {
        return Err(("config_invalid", format!("`{key}` is config-file content, not a live change: edit the file and restart")));
    }
    if !body.is_object() {
        return Err(("config_invalid", "the body must be a config document: `{ <group>: { <field>: value } }`".to_owned()));
    }
    let document = file::read_document(body, "the request body").map_err(|e| ("config_invalid", e.0))?;
    if document.values.is_empty() {
        return Err(("config_invalid", "the patch names no field".to_owned()));
    }
    let refused: Vec<String> =
        document.values.entries().filter(|(meta, _, _)| !meta.patchable).map(|(meta, _, _)| meta.file_key()).collect();
    if !refused.is_empty() {
        return Err((
            "config_field_not_patchable",
            format!(
                "{} cannot be changed while the server runs (not patchable); nothing in the patch was applied",
                refused.iter().map(|key| format!("`{key}`")).collect::<Vec<_>>().join(", ")
            ),
        ));
    }
    Ok(document.values)
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

/// The fields of `patch` that a flag or an env var of `running`'s start also
/// sets, each with the spelling that sets it: written to the file, they are
/// outranked again at the next start (spec config-v2/01 AC 6), so the
/// answer says so rather than promise a change that will not survive it.
pub fn shadowed_at_restart(patch: &Layer, running: &Config) -> Vec<String> {
    let sources = running.basis.sources();
    patch
        .entries()
        .filter_map(|(meta, family, _)| {
            let key = family.map_or_else(|| meta.file_key(), |family| meta.scoped_file_key(family));
            let by = sources.flags.get(meta, family).or_else(|| sources.env.get(meta, family))?;
            Some(format!("{key} (set by {})", by.spelling))
        })
        .collect()
}

/// Whether applying `patch` needs a model reload: any field in it does.
pub fn needs_reload(patch: &Layer) -> bool {
    patch.entries().any(|(meta, _, _)| meta.reload_required)
}

/// The checks a start makes on the thinking defaults, made again on the
/// loaded model for the ones a live change names (spec server/08, GitHub
/// #68): a default the template cannot honour, or a budget its tokenizer has
/// no forced close for, is refused rather than silently inert. Only for what
/// `patch` changes — a value the load already accepted is not judged again —
/// and the budget's check only on a model loaded from an artifact, as at
/// start: the placeholder has no tokenizer to close with.
fn check_thinking(server: &Server, next: &Config, patch: &Layer) -> Result<(), String> {
    let named = |name: &str| patch.entries().any(|(meta, _, _)| meta.group == "model" && meta.name == name);
    let active = server.active();
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

/// `PATCH /v1/config` — change the running configuration.
#[utoipa::path(
    patch,
    path = "/v1/config",
    tag = "config",
    operation_id = "patch_config",
    summary = "Change the running configuration",
    description = "The body is a partial config document in the config file's own shape — only the fields being changed, a `qwen38` / `qwen38flashnext` section for a family's own value. Each value is checked as the config file's would be; a field the loaded model's family cannot take, or one not patchable (the API key, the bind addresses, the Playground), refuses the **whole** patch, nothing in it applied. `switch.known_models` adds to the models already known.\n\nApplied one way, never two. When no field in the patch needs a model reload (the request timeout, the message policies, the thinking defaults, the model switch's own knobs) it takes effect at once, no model touched: `200`, with the configuration now running. When even one field does, the whole patch — its other fields included — goes through a reload of the loaded model with the patched configuration: the same switch `POST /v1/models/switch` runs, `202` at once, every other `/v1` route answering `503 model_switching` until it lands, and `GET /v1/models` reporting it. A reload that fails leaves the previous model serving with the previous configuration.\n\nA change is written to the config file in use once it has taken — at once, or when the reloaded model serves — keeping every other key in the file. With no config file in use it still applies, and `persisted: false` says why. `shadowed_at_restart` names the fields a flag or an env var also sets: they outrank the file at the next start.",
    request_body(content = ConfigDocument, description = "The fields to change, nested as the config file nests them."),
    responses(
        (status = 200, description = "Applied live.", body = PatchApplied),
        (status = 202, description = "A reload with the patched configuration began.", body = PatchApplied),
        (status = 400, description = "A field is unknown, not patchable (`config_field_not_patchable`), or wrong for the loaded model (`config_invalid`): nothing was applied.", body = ApiError),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 409, description = "A model switch is already under way (`switch_in_progress`), or the loaded model was not loaded from an artifact and cannot be reloaded (`reload_unavailable`).", body = ApiError),
        (status = 501, description = "This server has no running configuration, or no model loader to reload with.", body = ApiError),
        (status = 503, description = "The loaded model's first traversal has not run yet, or a model switch is under way.", body = ApiError),
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
    let shadowed = shadowed_at_restart(&patch, &running);
    if !needs_reload(&patch) {
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
            assert_eq!(document[meta.group].get(meta.name).is_some(), meta.visible, "{}", meta.file_key());
        }
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

    #[test]
    fn a_patch_refuses_profiles_unknown_fields_and_anything_not_patchable_whole() {
        assert_eq!(read_patch(&json!({ "profile": "x" })).unwrap_err().0, "config_invalid");
        assert!(read_patch(&json!({ "server": { "bnd": "x" } })).unwrap_err().1.contains("server.bnd"));
        assert!(read_patch(&json!({})).unwrap_err().1.contains("names no field"));
        let (code, message) =
            read_patch(&json!({ "server": { "request_timeout": 60, "api_key": "k", "bind": "x" } })).unwrap_err();
        assert_eq!(code, "config_field_not_patchable");
        assert!(message.contains("`server.api_key`") && message.contains("`server.bind`") && !message.contains("request_timeout"), "{message}");
        let live = read_patch(&json!({ "server": { "request_timeout": 60 }, "switch": { "drain_timeout": 0 } })).unwrap();
        assert!(!needs_reload(&live));
        let reload = read_patch(&json!({ "server": { "request_timeout": 60 }, "reuse": { "qwen38": { "kv_host_pool_bytes": "1G" } } })).unwrap();
        assert!(needs_reload(&reload));
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
}
