//! `GET /v1/models/catalog` (spec playground/05): every model this server
//! can switch to or fetch, with its state on disk — what a page needs to
//! offer a switch by id (`POST /v1/models/switch` with `model` alone) without
//! knowing a path on the server's machine.
//!
//! One row per catalog entry (built-in, then the operator's), then one per
//! known model (`switch.known_models`, the model the server started on) that
//! is not a catalog entry. An entry's state is what `model list` computes
//! ([`crate::download::command::list_rows`]: sizes only, never a hash); a
//! known model is `complete` when its artifact is a file, else `absent`. No
//! row carries a path: the page switches by id. Beside the rows, `disk` says
//! the free space under `download.path` (GitHub #453).

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use ignis_core::compute::{ModelFamily, ModelKind};
use serde::Serialize;
use utoipa::ToSchema;

use crate::api::ApiError;
use crate::config::field::scope;
use crate::download::catalog::CatalogLayer;
use crate::download::command::{file_size, free_under, lists_file, OnDisk};
use crate::download_http::DownloadStatus;
use crate::model_switch::Switcher;
use crate::{ModelStatus, Server};

/// Where a model is listed from.
#[derive(Serialize, ToSchema, Clone, Copy)]
#[serde(rename_all = "kebab-case")]
enum CatalogSource {
    /// An entry of the catalog compiled into the binary.
    BuiltIn,
    /// An entry of the operator's `download.catalog`.
    Operator,
    /// Not a catalog entry: in `switch.known_models`, or the model the server
    /// started on.
    Known,
}

impl From<CatalogLayer> for CatalogSource {
    fn from(layer: CatalogLayer) -> Self {
        match layer {
            CatalogLayer::BuiltIn => Self::BuiltIn,
            CatalogLayer::Operator => Self::Operator,
        }
    }
}

/// One model the server can switch to or fetch.
#[derive(Serialize, ToSchema)]
pub(crate) struct CatalogRow {
    /// The id the model is served under, and the one a switch names.
    id: String,
    /// `built-in` or `operator` (a catalog entry), or `known` (in
    /// `switch.known_models`, or the model the server started on, and not a
    /// catalog entry).
    source: CatalogSource,
    /// `complete`, `partial` or `absent` under `download.path`; `complete`
    /// too when the file the server would load the model from (a listed
    /// model's artifact, the one it started on) is there. A known model is
    /// `complete` when its artifact is a file, else `absent`.
    on_disk: OnDisk,
    /// An entry's total size, or the artifact's for a known model on disk;
    /// absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
    /// The scope name (`qwen38`, `qwen38flashnext`) the artifact's header
    /// names, when it is on disk; for the loaded model always the running
    /// family.
    #[serde(skip_serializing_if = "Option::is_none")]
    family: Option<&'static str>,
    /// What the model takes in and gives back (GitHub #448), as `GET
    /// /v1/models` names it: `text`, `image`, `speech` or `diarizer`. The
    /// artifact's header's when it is on disk, which wins (ADR 0047), the
    /// 27B's for a header of no family, as a load serves it; else what the
    /// catalog tells of the entry: its `kind` (a speech model, the diarizer)
    /// or its family's (ADR 0074). Absent when neither tells: an operator
    /// entry of no family not downloaded, a known model whose file is gone or
    /// has no header. For the loaded model, what it runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    /// This is the model serving now (the start model before any switch).
    /// Exactly one row is, except while nothing serves (a failed switch).
    loaded: bool,
    /// `POST /v1/models/switch` with this `model` alone would begin a switch
    /// to this row's own model now: the id resolves, its artifact is on
    /// disk, it fits the card's class, it is not the loaded one, and no
    /// switch is under way. A row above the card's class never is, though a
    /// switch naming its id may begin one to the entry that answers to it on
    /// this card (ADR 0064), which its `reason` names.
    switchable: bool,
    /// Whether this card can load it (ADR 0064): `false` for a catalog entry
    /// whose `card_class` is above the card's, whatever is on disk — a switch
    /// to it, or a download of it, is refused. Always `true` on the loaded
    /// row.
    available: bool,
    /// Why it is not available, naming its class and the card's — and the
    /// entry that serves a request naming it here, when one answers to its
    /// id. Only with `available: false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    /// Its download through the API (`POST /v1/models/download`, GitHub
    /// #376): `running` while one runs, `failed` when the last one failed
    /// (kept until the entry's next download, the entry complete on disk
    /// another way, or a restart), else `none`.
    /// Always `none` for a known model, which is never downloaded.
    download: CatalogDownload,
    /// Why the entry's last download failed. Only with `download: failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    download_reason: Option<String>,
    /// The bytes of it on disk now, partial files (`.part`) included: what a
    /// running download has fetched so far, of `bytes`. A known model's is
    /// its artifact's size, or 0.
    bytes_on_disk: u64,
    /// The id of the model the server started on, when it was loaded from
    /// one of this entry's files under `download.path` (GitHub #380),
    /// whatever id serves it: removed, the next start looks for it there
    /// again. Only on that row.
    #[serde(skip_serializing_if = "Option::is_none")]
    start_model: Option<String>,
    /// The ids `switch.known_models` names at one of this entry's files
    /// under `download.path`, the start model's apart: removed, a switch to
    /// any of them is refused until it is downloaded again. Only on such a
    /// row.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    known_models: Vec<String>,
}

/// Where a row's download stands (`CatalogRow::download`).
#[derive(Serialize, ToSchema, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum CatalogDownload {
    /// None runs, and the last one, if any, did not fail.
    None,
    /// One runs now.
    Running,
    /// The last one failed: `download_reason` says why.
    Failed,
}

/// The envelope of `GET /v1/models/catalog`, as `GET /v1/models`' is.
#[derive(Serialize, ToSchema)]
pub(crate) struct CatalogList {
    /// Always `list`.
    object: &'static str,
    /// The models, catalog entries first.
    data: Vec<CatalogRow>,
    /// The free space where the server downloads models (GitHub #453).
    /// Absent when it cannot be read, or the server has no catalog to
    /// download.
    #[serde(skip_serializing_if = "Option::is_none")]
    disk: Option<CatalogDisk>,
}

/// `download.path` and the bytes free on its volume (`CatalogList::disk`).
#[derive(Serialize, ToSchema)]
pub(crate) struct CatalogDisk {
    /// `download.path`, as the server holds it.
    path: String,
    /// The bytes free to the server on the volume `path` is on, read where a
    /// download checks them before it begins: at `path`, or at the nearest
    /// folder above it that exists, since the first download creates it.
    free_bytes: u64,
}

/// The list, as of now: the rows and the free space under `download.path`.
/// Reads the filesystem, as [`rows`] does.
fn list(server: &Server) -> CatalogList {
    let catalog = server.switcher.as_deref().and_then(Switcher::catalog);
    let disk = catalog.and_then(|catalog| {
        free_under(&catalog.dir).ok().map(|free_bytes| CatalogDisk { path: catalog.dir.display().to_string(), free_bytes })
    });
    CatalogList { object: "list", data: rows(server), disk }
}

/// The rows, as of now. Reads the filesystem (sizes, and the header of an
/// artifact that is there), so a caller on the async runtime moves it off
/// the workers.
fn rows(server: &Server) -> Vec<CatalogRow> {
    let active = server.active();
    let loaded_id = active.model_id();
    let switcher = server.switcher.as_deref();
    // A switch beginning now: serving, failed (the way out of a failure
    // is a switch), or unloaded (a switch loads, spec model-switch/02). Not
    // during one or a load, not before the warm-up has run.
    let status = server.status();
    let free = matches!(*status, ModelStatus::Serving | ModelStatus::Failed { .. } | ModelStatus::Unloaded);
    // After a failed switch, or an unload, nothing is resident: the engine
    // `active` still holds is the torn-down one, so no row is loaded and its
    // model is a way out — "Load" on the unloaded model loads it.
    let serving = !status.no_model_loaded();
    // A catalog entry's `on_disk` and `bytes_on_disk` both come from
    // `list_rows`, which reads each file once per listing: they never
    // disagree, as they did when a download landed files between two reads
    // (`absent` with bytes).
    let family_of = |path: &Path| crate::loader::artifact_family(path).ok().flatten();
    // GitHub #380: the model the server started on, and the files the known
    // models load from, which a removal of the entry listing them takes away.
    let started = switcher.and_then(Switcher::start);
    let known_models = switcher.map(Switcher::listed_models).unwrap_or_default();
    // GitHub #448: the kind the file a row would load from reads, when it is there.
    let kind_of = |path: &Path| crate::loader::artifact_kind(path);
    let row = |id: &str, source, on_disk: OnDisk, bytes, header_family: Option<ModelFamily>, header_kind: Option<ModelKind>, bytes_on_disk| {
        let entry = match source {
            CatalogSource::Known => None,
            CatalogSource::BuiltIn | CatalogSource::Operator => {
                switcher.and_then(Switcher::catalog).and_then(|catalog| catalog.catalog.entry(id).map(|entry| (entry, &catalog.dir)))
            }
        };
        // Matched by file, not by id: a model served under another id from
        // one of the entry's files loses them with it all the same.
        let loads_from = |path: &Path| entry.is_some_and(|(entry, dir)| lists_file(entry, dir, path));
        let start_model = started.as_ref().filter(|started| loads_from(&started.artifact)).map(|started| started.model.clone());
        let known_models: Vec<String> = known_models
            .iter()
            .filter(|(known, path)| started.as_ref().is_none_or(|started| started.model != **known) && loads_from(path))
            .map(|(known, _)| known.clone())
            .collect();
        let loaded = serving && id == loaded_id;
        // GitHub #448: the header, the authority where the file is there,
        // then what the catalog tells of the entry.
        let kind = if loaded { Some(active.kind()) } else { header_kind.or_else(|| entry.and_then(|(entry, _)| entry.model_kind())) };
        let reason = switcher.filter(|_| !loaded).and_then(|switcher| switcher.unavailable_reason(id));
        let status = match source {
            CatalogSource::Known => DownloadStatus::None,
            CatalogSource::BuiltIn | CatalogSource::Operator => {
                server.downloads.as_ref().map_or(DownloadStatus::None, |downloads| downloads.status(id, on_disk))
            }
        };
        let (download, download_reason) = match status {
            DownloadStatus::None => (CatalogDownload::None, None),
            DownloadStatus::Running => (CatalogDownload::Running, None),
            DownloadStatus::Failed(why) => (CatalogDownload::Failed, Some(why)),
        };
        CatalogRow {
            id: id.to_owned(),
            source,
            on_disk,
            bytes,
            family: if loaded { Some(scope(active.family)) } else { header_family.map(scope) },
            kind: kind.map(ModelKind::as_str),
            loaded,
            switchable: free && !loaded && switcher.is_some_and(|switcher| switcher.resolve(id).is_ok()),
            available: reason.is_none(),
            reason,
            download,
            download_reason,
            bytes_on_disk,
            start_model,
            known_models,
        }
    };

    let mut rows = Vec::new();
    let mut listed = switcher.map(|switcher| switcher.listed_models()).unwrap_or_default();
    // The model loaded by an explicit artifact is the server's whether or not
    // anyone listed it; the one row that is `loaded` must always exist.
    if let Some(source) = &active.source {
        listed.entry(source.model.clone()).or_insert_with(|| source.artifact.clone());
    }
    let mut seen = BTreeSet::new();
    if let Some(catalog) = switcher.and_then(|switcher| switcher.catalog()) {
        for found in crate::download::command::list_rows(&catalog.catalog, &catalog.dir, &file_size, &family_of) {
            seen.insert(found.entry.id.clone());
            // A listed model of the same id wins over `download.path`, as it
            // does for a switch (`Switcher::resolve`): its file is where the
            // server would load it from, and gone it is absent, whatever
            // `download.path` holds.
            let (on_disk, bytes, family, kind, landed) = match listed.get(&found.entry.id) {
                Some(artifact) => match file_size(artifact) {
                    Some(bytes) => (OnDisk::Complete, bytes, family_of(artifact), kind_of(artifact), bytes),
                    None => (OnDisk::Absent, found.entry.total_bytes(), None, None, 0),
                },
                None => {
                    let kind = kind_of(&found.entry.artifact_path(&catalog.dir));
                    (found.on_disk, found.entry.total_bytes(), found.family, kind, found.bytes_on_disk)
                }
            };
            rows.push(row(&found.entry.id, found.entry.layer.into(), on_disk, Some(bytes), family, kind, landed));
        }
    }
    for (id, artifact) in listed.iter().filter(|(id, _)| !seen.contains(*id)) {
        let bytes = file_size(artifact);
        let on_disk = if bytes.is_some() { OnDisk::Complete } else { OnDisk::Absent };
        let family = bytes.and_then(|_| family_of(artifact));
        let kind = bytes.and_then(|_| kind_of(artifact));
        rows.push(row(id, CatalogSource::Known, on_disk, bytes, family, kind, bytes.unwrap_or(0)));
    }
    // A loaded model with no artifact at all (a start built in process).
    if serving && !rows.iter().any(|row| row.loaded) {
        rows.push(row(&loaded_id, CatalogSource::Known, OnDisk::Absent, None, None, None, 0));
    }
    rows
}

/// `GET /v1/models/catalog` — the models this server can switch to or fetch.
#[utoipa::path(
    get,
    path = "/v1/models/catalog",
    tag = "models",
    operation_id = "list_model_catalog",
    summary = "The models this server can switch to or fetch",
    description = "One row per model the server knows of: every catalog entry (the built-in ones, then the operator's `download.catalog`), then every known model (`switch.known_models`, and the model the server started on) that is not a catalog entry. Each row says where the model is from (`source`), how much of it is there (`on_disk`: `complete`, `partial` or `absent` under `download.path`, or `complete` when the artifact a listed model or the start names is a file — file sizes only, never a hash, as `ignis-server model list` computes it; a known model is `complete` when its artifact is a file), its size, the model family its artifact's header names, its model kind (`kind`: `text`, `image`, `speech` or `diarizer`, as `GET /v1/models` names it: the artifact's header's when the file is there, which wins, else what the catalog tells of the entry, its stated `kind` or its family's; absent when neither tells), whether it is the model serving now (`loaded`: exactly one row, none while a failed switch has left nothing serving) and whether `POST /v1/models/switch` with this `model` alone would begin a switch to this row's own model now (`switchable`: the id resolves, the artifact is on disk, it fits the card's class, it is not the loaded one, and no switch is under way). A catalog entry whose `card_class` is above this card's (ADR 0064) is `available: false`, with the `reason` naming its class and the card's, and the entry that serves a request naming it here when one answers to its id; it is never switchable itself — a switch naming its id switches to that entry, when there is one — and `model download` refuses it. Every other row is `available: true`. No row carries a path: switch by id. A model not on disk is fetched with `ignis-server model download <id>`, or in the background with `POST /v1/models/download`; a switch never starts a download. Each row carries `bytes_on_disk`, what of it is there now, partial files included, and a catalog entry's `download`: `running` while that download runs (reading the catalog again shows its progress), `failed` with its `download_reason` once one failed, until the entry's next download, the entry complete on disk another way (`ignis-server model download`), or a restart, else `none`. The row of an entry the server started on from one of its files under `download.path` names that model in `start_model`, and the row of an entry `switch.known_models` names at one of its files there lists those ids in `known_models`, whatever id each is served under: what `POST /v1/models/remove` of it takes away. Beside the rows the list carries `disk`, `{path, free_bytes}`: `download.path` as the server holds it and the bytes free to the server on its volume, read where a download checks them before it begins (at the nearest folder above it that exists, since the first download creates it); absent when they cannot be read.\n\nAnswers during a model switch and while one has failed, like `GET /v1/models`.",
    responses(
        (status = 200, description = "The models, catalog entries first.", body = CatalogList),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 503, description = "The loaded model's first traversal has not run yet (`server_not_ready`).", body = ApiError),
    ),
)]
pub(crate) async fn get_models_catalog(State(server): State<Arc<Server>>) -> Json<CatalogList> {
    let server = Server::clone(&server);
    Json(tokio::task::spawn_blocking(move || list(&server)).await.expect("listing the catalog does not panic"))
}
