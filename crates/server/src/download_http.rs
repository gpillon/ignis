//! `POST /v1/models/download` (GitHub #376, spec #374, ADR 0071): a
//! **Download** of a catalog entry that the server runs in the background
//! while it keeps serving — the Server view's Download button — and the
//! state `GET /v1/models/catalog` shows for each entry while it runs and
//! after it failed.
//!
//! The transfer is the one a start and `model download` run
//! ([`crate::download::Downloader::start`]): resumed from a `.part`,
//! verified against the pins, the token sent to the endpoint only. One
//! download runs at a time. What runs, and the last failure of each entry,
//! are kept in memory ([`Downloads`]): nothing is written about them, and
//! nothing resumes at a restart. A switch never downloads.
//!
//! `POST /v1/models/download/cancel` (GitHub #377) stops the running
//! download — the Server view's Cancel — and keeps what it fetched: the
//! entry reads partial, and its next download, the row's Resume, continues
//! from those files.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::api::{error_response, error_response_naming, ApiError};
use crate::config::{ApiKey, Config};
use crate::download::command::{file_size, on_disk, room_for, OnDisk};
use crate::download::{CatalogEntry, Download, DownloadError, Downloader, Ended, EntryLock, Holder, NotLocked};
use crate::Server;

/// The downloads a server runs through the API, and how they are fetched:
/// shared by every clone of the server ([`Server::with_downloads`]).
pub struct Downloads {
    /// `download.endpoint`.
    endpoint: String,
    /// The token [`crate::download::bearer_token`] chose at start.
    token: Option<ApiKey>,
    /// `download.enabled`: off, the server never uses the network.
    enabled: bool,
    /// `download.from_ui`: off, the API downloads nothing.
    from_ui: bool,
    /// `download.free_margin`: what a download must leave free beyond the
    /// bytes it still misses (GitHub #378).
    free_margin: u64,
    /// The download under way, or the last one until something settles it
    /// ([`Downloads::settle`]).
    running: Mutex<Option<Download>>,
    /// Why each entry's last download failed, by id: kept until the entry's
    /// next download begins, the entry is complete on disk another way
    /// (`model download`), or the server restarts.
    failures: Mutex<BTreeMap<String, String>>,
}

/// Where a catalog entry's downloads stand, for its catalog row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadStatus {
    /// None runs, and the last one, if any, did not fail.
    None,
    /// One runs now.
    Running,
    /// The last one failed, for this reason.
    Failed(String),
}

impl Downloads {
    /// How `config` says to download, with `hf_token` the `HF_TOKEN` of the
    /// server's environment, read once at start as the start-time fetch
    /// reads it.
    pub fn of(config: &Config, hf_token: Option<&str>) -> Self {
        Self {
            endpoint: config.download_endpoint.clone(),
            token: crate::download::bearer_token(&config.download_endpoint, config.download_token.as_ref(), hf_token),
            enabled: config.model_download,
            from_ui: config.settings().download.from_ui,
            free_margin: config.settings().download.free_margin,
            running: Mutex::new(None),
            failures: Mutex::new(BTreeMap::new()),
        }
    }

    /// The download running now, if one is.
    pub fn running(&self) -> Option<Download> {
        lock(&self.running).as_ref().filter(|download| download.ended().is_none()).cloned()
    }

    /// Where the downloads of the entry `id` stand, `on_disk` as its row
    /// reads it: running, failed with the reason of the last one, or
    /// neither. A download that has ended reads as it ended, whether or not
    /// it has been settled yet. An entry complete on disk reads no failure: one
    /// kept from before it was completed another way (`model download`, a
    /// copy) is forgotten, so it never outlives the entry it was about.
    pub fn status(&self, id: &str, on_disk: OnDisk) -> DownloadStatus {
        let complete = on_disk == OnDisk::Complete;
        let running = lock(&self.running);
        if let Some(download) = running.as_ref().filter(|download| download.entry().id == id) {
            match download.ended() {
                None => return DownloadStatus::Running,
                Some(Ended::Failed(err)) if !complete => return DownloadStatus::Failed(err.to_string()),
                Some(_) => return DownloadStatus::None,
            }
        }
        let mut failures = lock(&self.failures);
        if complete {
            failures.remove(id);
            return DownloadStatus::None;
        }
        failures.get(id).map_or(DownloadStatus::None, |reason| DownloadStatus::Failed(reason.clone()))
    }

    /// Fetch `entry` into `dir` in the background, unless a download runs —
    /// then that one, which is refused — or another fetch of the entry holds
    /// its lock: `model download`, or another server on the same folder. The
    /// entry's last failure is forgotten as this one begins.
    fn begin(self: &Arc<Self>, downloader: Downloader, entry: CatalogEntry, dir: PathBuf) -> Result<Download, NotBegun> {
        let mut running = lock(&self.running);
        self.settle(&mut running);
        if let Some(busy) = running.as_ref() {
            return Err(NotBegun::Busy(busy.clone()));
        }
        let id = entry.id.clone();
        let download = downloader.start(entry, dir).map_err(NotBegun::Locked)?;
        lock(&self.failures).remove(&id);
        *running = Some(download.clone());
        drop(running);
        // What becomes of it is recorded as soon as it ends, whether or not
        // anyone reads the catalog.
        let (downloads, ended) = (Arc::clone(self), download.clone());
        tokio::spawn(async move {
            ended.wait().await;
            downloads.settle(&mut lock(&downloads.running));
        });
        Ok(download)
    }

    /// Clear the slot of a download that has ended, keeping a failure's
    /// reason for its entry and saying it in the log, as a cancel is said
    /// there too. One that still runs stays.
    fn settle(&self, running: &mut Option<Download>) {
        let Some(ended) = running.as_ref().and_then(Download::ended) else {
            return;
        };
        let download = running.take().expect("the download that ended");
        // GitHub #377: a cancel says so, and what it kept for the resume.
        if let Ended::Cancelled = ended {
            tracing::info!(
                name: "ignis.model.download_cancelled",
                model = %download.entry().id,
                repo = %download.entry().repo,
                endpoint = %self.endpoint,
                bytes = download.entry().total_bytes(),
                bytes_on_disk = download.bytes_on_disk(),
                "the download was cancelled; its partial files stay, and the entry's next download resumes from them"
            );
        }
        if let Ended::Failed(err) = ended {
            tracing::warn!(
                name: "ignis.model.download_failed",
                model = %download.entry().id,
                repo = %download.entry().repo,
                endpoint = %self.endpoint,
                error = %err,
                "the download did not land; GET /v1/models/catalog shows why until the entry's next download"
            );
            lock(&self.failures).insert(download.entry().id.clone(), err.to_string());
        }
    }

    /// Keep `reason` as the entry `id`'s last failure: a download refused
    /// for want of room before it began (GitHub #378), which its catalog row
    /// then shows as a failed one's, until the entry's next download or a
    /// restart.
    fn refused(&self, id: &str, reason: &str) {
        lock(&self.failures).insert(id.to_owned(), reason.to_owned());
    }

    /// `download.from_ui`: off, the API neither downloads nor removes
    /// (GitHub #380, ADR 0071).
    pub(crate) fn from_ui(&self) -> bool {
        self.from_ui
    }

    /// Forget the entry `id`'s last failure: its files were removed (GitHub
    /// #380), and its row reads as one never downloaded.
    pub(crate) fn forget(&self, id: &str) {
        lock(&self.failures).remove(id);
    }
}

/// Why [`Downloads::begin`] began nothing.
enum NotBegun {
    /// A download runs: this one.
    Busy(Download),
    /// The entry's lock was not taken: another fetch holds it, or it could
    /// not be opened.
    Locked(NotLocked),
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The body of `POST /v1/models/download`.
#[derive(Deserialize, ToSchema)]
pub(crate) struct DownloadRequest {
    /// The catalog entry to download, by the id it is served under (`GET
    /// /v1/models/catalog`'s `id`).
    model: String,
}

/// A download that began.
#[derive(Serialize, ToSchema)]
pub(crate) struct DownloadAccepted {
    /// Always `downloading`.
    status: &'static str,
    /// The entry it downloads.
    model: String,
    /// The entry's total size.
    bytes: u64,
    /// What of it was already under `download.path` as it began: a partial
    /// download it resumes.
    bytes_on_disk: u64,
}

/// `POST /v1/models/download` — download a catalog entry in the background.
#[utoipa::path(
    post,
    path = "/v1/models/download",
    tag = "models",
    operation_id = "download_model",
    summary = "Download a catalog entry in the background",
    description = "Fetches every file of the catalog entry `model` names (`GET /v1/models/catalog` lists them: the built-in entries and the operator's `download.catalog`) into `download.path`, from `download.endpoint`, with the same transfer as `ignis-server model download`: each file resumed from its `.part`, checked against its pinned size and SHA-256 before it is renamed into place, the token (`download.token`, else `HF_TOKEN` for huggingface.co, as the server read them at start) sent to the endpoint only. Nothing but a catalog entry is downloaded, and the request never carries a token.\n\nAnswers `202` at once; the download runs in the background while the loaded model keeps serving. `GET /v1/models/catalog` follows it: the entry's row says `download: running` with its `bytes_on_disk`, then, once every file verified, `on_disk: complete` and `download: none` — a servable model, which `POST /v1/models/switch` loads; the server never switches to it by itself. A download that fails leaves `download: failed` and its `download_reason` on the row until the entry's next download, the entry complete on disk another way (`ignis-server model download`), or a restart; a `401` or `403` from the endpoint names the token and where the server reads it. One download runs at a time. A partial entry's download resumes it. An entry `ignis-server model download` is fetching into the same folder is refused until it ends: each fetch holds the entry's lock, `<artifact>.lock` in `download.path`, for as long as it runs. Under the same API key as `PATCH /v1/config` (ADR 0071); `download.enabled` false or `download.from_ui` false refuses it.",
    request_body = DownloadRequest,
    responses(
        (status = 202, description = "The download began.", body = DownloadAccepted),
        (status = 400, description = "`model` is empty.", body = ApiError),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 403, description = "Downloading through the API is off (`download_disabled`): `download.enabled` is false, so the server never uses the network, or `download.from_ui` is false; the message names which.", body = ApiError),
        (status = 404, description = "`model` is not a catalog entry (`model_not_found`): a known model is a path, and nothing but a catalog entry is downloaded.", body = ApiError),
        (status = 409, description = "A download already runs (`download_in_progress`, naming it): downloads are one at a time, never queued. Or the entry's `card_class` is above this card's (`model_above_card_class`, ADR 0064), or its `generation` newer than the card's GPU generation, alone or with its class (`model_above_gpu_generation`, ADR 0079), or every file of it is already on disk (`model_already_downloaded`). Or another fetch of the entry into `download.path` holds its lock (`download_locked`): `ignis-server model download`, or another server on the same folder, which the message names with its process id; one writer per entry.", body = ApiError),
        (status = 500, description = "The download could not begin (`download_failed`): the HTTP client could not be built, or the entry's lock file under `download.path` could not be opened; the message says which. Nothing was fetched.", body = ApiError),
        (status = 501, description = "This server has no catalog to download from (`download_unavailable`).", body = ApiError),
        (status = 503, description = "The loaded model's first traversal has not run yet (`server_not_ready`), or a model switch is under way (`model_switching`).", body = ApiError),
        (status = 507, description = "The volume of `download.path` has less free space than the entry's bytes still missing plus `download.free_margin` (`download_no_room`): nothing was written, and the message names the bytes needed and the bytes free. The entry's row keeps it as its `download_reason`, `download: failed`, until its next download, the entry complete on disk another way, or a restart.", body = ApiError),
    ),
)]
pub(crate) async fn post_model_download(State(server): State<Arc<Server>>, Json(req): Json<DownloadRequest>) -> Response {
    if req.model.trim().is_empty() {
        return crate::api::bad_request_param("`model` must name the catalog entry to download", "model");
    }
    let switcher = server.switcher.as_deref();
    let (Some(downloads), Some(catalog)) = (server.downloads.as_ref(), switcher.and_then(|switcher| switcher.catalog())) else {
        return error_response(StatusCode::NOT_IMPLEMENTED, "server_error", "download_unavailable", "this server has no catalog to download models from");
    };
    if !downloads.enabled {
        return disabled("downloading is off on this server: download.enabled is false, so it never uses the network");
    }
    if !downloads.from_ui {
        return disabled(&format!(
            "downloading through the API is off on this server (download.from_ui is false); `ignis-server model download {}` on the server's machine still fetches it",
            req.model.trim()
        ));
    }
    let Some(entry) = catalog.catalog.entry(&req.model) else {
        return error_response_naming(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "model_not_found",
            format!("unknown model: {} is not a catalog entry, and only catalog entries are downloaded; GET /v1/models/catalog lists them", req.model.trim()),
            Some("model"),
        );
    };
    if let Some((code, reason)) = switcher.and_then(|switcher| switcher.unavailable(&entry.id)) {
        return error_response_naming(
            StatusCode::CONFLICT,
            "invalid_request_error",
            code,
            format!("{reason}: it is not downloaded for this card"),
            Some("model"),
        );
    }
    if on_disk(entry, &catalog.dir, &file_size) == OnDisk::Complete {
        return error_response_naming(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "model_already_downloaded",
            format!("{} is already on disk under download.path; POST /v1/models/switch with it loads it", entry.id),
            Some("model"),
        );
    }
    // GitHub #378: room for what is still missing plus download.free_margin,
    // read before a byte is written. The row keeps the refusal as the entry's
    // failure, so whoever asked sees it where they asked. While a download
    // runs, `begin` refuses first: one of the entry itself must not leave a
    // refusal behind on a row it then completes.
    if downloads.running().is_none()
        && let Err(reason) = room_for(entry, &catalog.dir, downloads.free_margin)
    {
        // GitHub #379: so does another fetch writing the entry, `model
        // download` or another server: it is refused for that instead, and
        // leaves no refusal for room behind on the row it then completes.
        if let Err(NotLocked::Held(err)) = EntryLock::take(entry, &catalog.dir, Holder::Server) {
            return locked(err);
        }
        downloads.refused(&entry.id, &reason);
        return error_response(StatusCode::INSUFFICIENT_STORAGE, "server_error", "download_no_room", reason);
    }
    let downloader = match Downloader::new(&downloads.endpoint, downloads.token.clone()) {
        Ok(downloader) => downloader,
        Err(err) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "download_failed", err.to_string()),
    };
    let download = match downloads.begin(downloader, entry.clone(), catalog.dir.clone()) {
        Ok(download) => download,
        Err(NotBegun::Busy(busy)) => {
            return error_response(
                StatusCode::CONFLICT,
                "invalid_request_error",
                "download_in_progress",
                format!("a download of {} is already running; downloads are one at a time, never queued", busy.entry().id),
            )
        }
        Err(NotBegun::Locked(NotLocked::Held(err))) => return locked(err),
        Err(NotBegun::Locked(NotLocked::Failed(err))) => {
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "download_failed", err.to_string())
        }
    };
    (
        StatusCode::ACCEPTED,
        Json(DownloadAccepted { status: "downloading", model: entry.id.clone(), bytes: entry.total_bytes(), bytes_on_disk: download.bytes_on_disk() }),
    )
        .into_response()
}

/// `403 download_disabled`, saying which switch is off.
fn disabled(message: &str) -> Response {
    error_response(StatusCode::FORBIDDEN, "invalid_request_error", "download_disabled", message)
}

/// `409 download_locked`: another fetch of the entry holds its lock (GitHub
/// #379), as `err` says.
pub(crate) fn locked(err: DownloadError) -> Response {
    error_response_naming(StatusCode::CONFLICT, "invalid_request_error", "download_locked", err.to_string(), Some("model"))
}

/// A download that was cancelled (GitHub #377).
#[derive(Serialize, ToSchema)]
pub(crate) struct DownloadCancelled {
    /// Always `cancelled`.
    status: &'static str,
    /// The entry it was downloading.
    model: String,
    /// The entry's total size.
    bytes: u64,
    /// What of it stays under `download.path`, partial files included: what
    /// the entry's next download resumes from.
    bytes_on_disk: u64,
}

/// `POST /v1/models/download/cancel` — stop the running download.
#[utoipa::path(
    post,
    path = "/v1/models/download/cancel",
    tag = "models",
    operation_id = "cancel_model_download",
    summary = "Stop the running download, keeping its partial files",
    description = "Stops the download `POST /v1/models/download` began, the one at a time that runs, and answers once it has stopped. What it fetched stays under `download.path`: every file it verified at its name, and the `.part` of the file it was writing with every byte that arrived. `GET /v1/models/catalog` then shows the entry `on_disk: partial` with its `bytes_on_disk` and `download: none`, and the entry's next `POST /v1/models/download` resumes from those files, with `Range`, and verifies them against the pins. Nothing resumes by itself, at a restart either. The request has no body. Logs `ignis.model.download_cancelled`. A cancel never touches the loaded model, so it is answered during a model switch too. Under the same API key as `PATCH /v1/config` (ADR 0071).",
    responses(
        (status = 200, description = "The download stopped; its partial files stay.", body = DownloadCancelled),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 409, description = "No download is running (`download_not_running`): none began, or the one that ran had already landed or failed, which the message says.", body = ApiError),
        (status = 501, description = "This server has no catalog to download from (`download_unavailable`).", body = ApiError),
        (status = 503, description = "The loaded model's first traversal has not run yet (`server_not_ready`).", body = ApiError),
    ),
)]
pub(crate) async fn post_model_download_cancel(State(server): State<Arc<Server>>) -> Response {
    let Some(downloads) = server.downloads.as_ref() else {
        return error_response(StatusCode::NOT_IMPLEMENTED, "server_error", "download_unavailable", "this server has no catalog to download models from");
    };
    let Some(download) = downloads.running() else {
        return not_running("no download is running; GET /v1/models/catalog shows each entry's state".to_owned());
    };
    download.cancel();
    let id = &download.entry().id;
    match download.wait().await {
        Ended::Cancelled => (
            StatusCode::OK,
            Json(DownloadCancelled { status: "cancelled", model: id.clone(), bytes: download.entry().total_bytes(), bytes_on_disk: download.bytes_on_disk() }),
        )
            .into_response(),
        // It ended on its own before the cancel reached it.
        Ended::Fetched(_) => not_running(format!("the download of {id} landed before the cancel reached it; POST /v1/models/switch with it loads it")),
        Ended::Failed(err) => not_running(format!("the download of {id} had already failed before the cancel reached it: {err}")),
    }
}

/// `409 download_not_running`: nothing to cancel.
fn not_running(message: String) -> Response {
    error_response(StatusCode::CONFLICT, "invalid_request_error", "download_not_running", message)
}
