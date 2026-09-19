//! Fetching the model when it is not on disk (GitHub #234, spec
//! `.scratch/model-download/specs/01-model-download.md`).
//!
//! A server started with no `.ninfer` artifact used to fall back to the
//! placeholder template and `MockCompute` without a word about where the real
//! model comes from. The artifact is published — the owner's own Hugging Face
//! repo, the one `models/*.README.md` documents — so the server can fetch it
//! itself: it asks first when somebody is there to answer (stdin is a TTY),
//! and just downloads when nobody is (a container, a daemon, CI).
//!
//! Two seams, so nothing here needs the network to be tested:
//!
//! - [`artifact_source`] is the **decision**: pure, given an `exists` probe
//!   and whatever the operator answered. Every cell of the spec's table is a
//!   unit test below.
//! - [`Downloader`] is the **transfer**, with an injectable base URL, driven
//!   in tests against a local `axum` server (`tests/model_download.rs`).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

/// A model the server knows how to fetch: where it is published, what the
/// files are called, and what the artifact must hash to.
///
/// The byte count and digest are **pinned here**, never read from the repo:
/// the published `.sha256` sits in the same trust domain as the artifact it
/// describes, so it can attest nothing about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelEntry {
    /// The model id `--model` names (what `GET /v1/models` reports).
    pub model: &'static str,
    /// The Hugging Face repo, `<owner>/<name>`.
    pub repo: &'static str,
    /// The `.ninfer` container's file name, kept verbatim on disk so a
    /// `hf download --local-dir models` of the same repo is the same file.
    pub artifact_file: &'static str,
    /// The provenance record next to it (ADR 0002) — the loader refuses a
    /// load without one, so it is part of the download, not an extra.
    pub sidecar_file: &'static str,
    /// The artifact's exact size in bytes.
    pub bytes: u64,
    /// The artifact's SHA-256, lowercase hex.
    pub sha256: &'static str,
}

impl ModelEntry {
    /// Where this model's artifact lives under `dir` (flat, one file per
    /// model — what `hf download <repo> <file> --local-dir <dir>` produces).
    pub fn artifact_path(&self, dir: &Path) -> PathBuf {
        dir.join(self.artifact_file)
    }

    /// Where this model's sidecar lives under `dir`.
    pub fn sidecar_path(&self, dir: &Path) -> PathBuf {
        dir.join(self.sidecar_file)
    }

    /// The artifact's size in GiB — for the question and the log line, never
    /// for a decision.
    pub fn gib(&self) -> f64 {
        (self.bytes as f64) / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Every model the server can fetch. One entry today: the v1 specialization
/// (`CONTEXT.md`), the DFlash2-grafted v2 image of the `nvfp4full` profile.
pub const REGISTRY: &[ModelEntry] = &[ModelEntry {
    model: crate::config::DEFAULT_MODEL,
    repo: "gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer",
    artifact_file: "qwen3_8_27b_nvfp4full-v2.ninfer",
    sidecar_file: "qwen3_8_27b_nvfp4full-v2.ninfer.graft.json",
    bytes: 19_406_942_468,
    sha256: "abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af",
}];

/// The registry entry for `model`, if the server knows how to fetch it.
/// Matched case-insensitively: a model id is a name, not a checksum.
pub fn registry_entry(model: &str) -> Option<&'static ModelEntry> {
    REGISTRY
        .iter()
        .find(|entry| entry.model.eq_ignore_ascii_case(model.trim()))
}

/// Where the server's artifact comes from, decided once at start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactSource {
    /// Load this file (it is on disk, or the operator named it and owns the
    /// consequences of it not being).
    Use(PathBuf),
    /// Fetch `entry` into `dir` first, then load what lands there.
    Download {
        entry: &'static ModelEntry,
        dir: PathBuf,
    },
    /// No artifact: the placeholder template and `MockCompute`, as before.
    Placeholder(PlaceholderReason),
}

/// Why a start carries no artifact — one warn line each, so the operator is
/// never left guessing which of these happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderReason {
    /// This binary was built without `--features cuda`: it could not run the
    /// weights it downloaded, so it never downloads them.
    NotSupported,
    /// `--model` names a model the registry has no entry for.
    UnknownModel,
    /// `--no-model-download` / `IGNIS_MODEL_DOWNLOAD=false`.
    DownloadsDisabled,
    /// The operator was asked and said no.
    Declined,
}

impl PlaceholderReason {
    /// The reason as a log field value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotSupported => "not-supported",
            Self::UnknownModel => "unknown-model",
            Self::DownloadsDisabled => "downloads-disabled",
            Self::Declined => "declined",
        }
    }
}

/// What the config says about fetching a missing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownloadSettings<'a> {
    /// `--artifact` / `IGNIS_ARTIFACT`, when the operator named one.
    pub artifact: Option<&'a Path>,
    /// `--model` / `IGNIS_MODEL` — the registry key.
    pub model: &'a str,
    /// `--model-download` / `--no-model-download`.
    pub enabled: bool,
    /// `--model-download-path`: where a downloaded artifact lands.
    pub dir: &'a Path,
    /// Whether this binary can run a model at all (built with `cuda`).
    pub supported: bool,
}

/// Decide where the artifact comes from (the spec's table, in one pure
/// function).
///
/// `exists` probes the filesystem; `ask` puts the question to the operator
/// and returns `Some(answer)`, or `None` when there is nobody to ask (stdin
/// is not a TTY). It is called only when the answer can still change the
/// outcome, so a start that already has its model is never interrupted by a
/// question.
pub fn artifact_source(
    settings: &DownloadSettings<'_>,
    exists: impl Fn(&Path) -> bool,
    ask: impl FnOnce(&ModelEntry, &Path) -> Option<bool>,
) -> ArtifactSource {
    // The operator's own path wins whatever else is true, missing or not:
    // `main` refuses the start on a path it cannot open, and a typo must stay
    // a typo rather than become a 19 GB download somewhere else.
    if let Some(artifact) = settings.artifact {
        return ArtifactSource::Use(artifact.to_path_buf());
    }
    if !settings.supported {
        return ArtifactSource::Placeholder(PlaceholderReason::NotSupported);
    }
    let Some(entry) = registry_entry(settings.model) else {
        return ArtifactSource::Placeholder(PlaceholderReason::UnknownModel);
    };
    let path = entry.artifact_path(settings.dir);
    // Already downloaded (by this server, by `hf download`, or by the shared
    // model store this directory points at): load it, never fetch it again.
    if exists(&path) {
        return ArtifactSource::Use(path);
    }
    if !settings.enabled {
        return ArtifactSource::Placeholder(PlaceholderReason::DownloadsDisabled);
    }
    let download = ArtifactSource::Download {
        entry,
        dir: settings.dir.to_path_buf(),
    };
    match ask(entry, &path) {
        // Nobody to ask: a container or a daemon cannot answer a question,
        // and coming up on the placeholder would be the wrong surprise.
        None | Some(true) => download,
        Some(false) => ArtifactSource::Placeholder(PlaceholderReason::Declined),
    }
}

// ── the transfer ────────────────────────────────────────────────────────────

/// Where the artifacts are published. Overridden only by tests, which drive
/// the transfer against a local server instead of the internet.
pub const HUGGINGFACE: &str = "https://huggingface.co";

/// How long the connection alone may take. The transfer itself is deliberately
/// **not** capped: 19.4 GB over a slow line is a long download, not a hung one.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Read in this much of the `.part` at a time when a resumed transfer replays
/// it through the hasher.
const RESUME_CHUNK: usize = 1 << 20;

/// How much of the body must arrive between two progress lines.
const PROGRESS_STEP: f64 = 0.05;

/// A transfer that did not produce a verified artifact. Never a panic, and
/// never a partially-written file the next start would trust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadError(pub String);

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DownloadError {}

impl DownloadError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Fetches a [`ModelEntry`]'s files into a directory.
///
/// The base URL is injectable so the whole path — resume, truncation, a body
/// that is not what it claims to be — is covered by tests against a loopback
/// server (`tests/model_download.rs`).
pub struct Downloader {
    client: reqwest::Client,
    base: String,
}

impl Downloader {
    /// The real one: Hugging Face.
    pub fn huggingface() -> Result<Self, DownloadError> {
        Self::with_base(HUGGINGFACE)
    }

    /// One pointed at `base` (`https://host`, no trailing slash needed).
    pub fn with_base(base: impl Into<String>) -> Result<Self, DownloadError> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .user_agent(concat!("ignis/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|err| DownloadError::new(format!("HTTP client: {err}")))?;
        let base = base.into().trim_end_matches('/').to_owned();
        Ok(Self { client, base })
    }

    /// The URL a repo file resolves to (Hugging Face's own route shape).
    fn url(&self, repo: &str, file: &str) -> String {
        format!("{}/{repo}/resolve/main/{file}", self.base)
    }

    /// Fetch `entry` into `dir` and return the artifact's path.
    ///
    /// The sidecar comes first: it is one small request, the loader refuses a
    /// load without it, and a repo that cannot serve it fails in a second
    /// instead of an hour. The artifact then streams to `<name>.part`, is
    /// hashed as it is written, and is renamed into place only once its
    /// length and digest are the pinned ones.
    pub async fn fetch(&self, entry: &ModelEntry, dir: &Path) -> Result<PathBuf, DownloadError> {
        tokio::fs::create_dir_all(dir).await.map_err(|err| {
            DownloadError::new(format!("cannot create {}: {err}", dir.display()))
        })?;
        self.fetch_sidecar(entry, &entry.sidecar_path(dir)).await?;
        let artifact = entry.artifact_path(dir);
        self.fetch_artifact(entry, &artifact).await?;
        Ok(artifact)
    }

    /// The provenance record (ADR 0002) — small enough to buffer whole, and
    /// written through a `.part` so an interrupted write never leaves a
    /// truncated record the loader would parse.
    async fn fetch_sidecar(&self, entry: &ModelEntry, path: &Path) -> Result<(), DownloadError> {
        let url = self.url(entry.repo, entry.sidecar_file);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|err| DownloadError::new(format!("GET {url}: {err}")))?;
        let status = response.status();
        if !status.is_success() {
            return Err(DownloadError::new(format!(
                "GET {url}: {} {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("")
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|err| DownloadError::new(format!("GET {url}: {err}")))?;
        let part = part_path(path);
        tokio::fs::write(&part, &bytes)
            .await
            .map_err(|err| DownloadError::new(format!("write {}: {err}", part.display())))?;
        tokio::fs::rename(&part, path)
            .await
            .map_err(|err| DownloadError::new(format!("rename {}: {err}", part.display())))?;
        tracing::info!(
            name: "ignis.model.sidecar_downloaded",
            file = entry.sidecar_file,
            bytes = bytes.len(),
            path = %path.display(),
            "provenance record fetched"
        );
        Ok(())
    }

    /// The weights. Resumes an earlier `.part` when there is one, verifies
    /// length and digest, and only then renames.
    async fn fetch_artifact(&self, entry: &ModelEntry, path: &Path) -> Result<(), DownloadError> {
        let part = part_path(path);
        let url = self.url(entry.repo, entry.artifact_file);
        // What an earlier attempt left. More bytes than the pin names is not
        // a resume point but a different file, so it starts over.
        let mut have = match tokio::fs::metadata(&part).await {
            Ok(meta) if meta.len() < entry.bytes => meta.len(),
            Ok(_) => 0,
            Err(_) => 0,
        };
        let mut hasher = Sha256::new();
        if have > 0 {
            replay_into_hasher(&part, have, &mut hasher).await?;
        }

        let mut request = self.client.get(&url);
        if have > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut response = request
            .send()
            .await
            .map_err(|err| DownloadError::new(format!("GET {url}: {err}")))?;
        let status = response.status();
        if !status.is_success() {
            return Err(DownloadError::new(format!(
                "GET {url}: {} {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("")
            )));
        }
        // A `200` to a ranged request is the whole body from zero: appending
        // it to what is already there would produce a file that is neither.
        if have > 0 && status != reqwest::StatusCode::PARTIAL_CONTENT {
            tracing::warn!(
                name: "ignis.model.resume_unsupported",
                file = entry.artifact_file,
                have,
                "the source ignored Range — starting the transfer over"
            );
            have = 0;
            hasher = Sha256::new();
        }

        let mut file = open_part(&part, have).await?;
        tracing::info!(
            name: "ignis.model.download_started",
            model = entry.model,
            repo = entry.repo,
            file = entry.artifact_file,
            url = %url,
            path = %path.display(),
            total_bytes = entry.bytes,
            resume_from = have,
            "fetching the model"
        );

        let started = Instant::now();
        let mut written = have;
        let mut next_log = written as f64 / entry.bytes.max(1) as f64 + PROGRESS_STEP;
        loop {
            let chunk = response
                .chunk()
                .await
                .map_err(|err| DownloadError::new(format!("GET {url}: {err}")))?;
            let Some(chunk) = chunk else { break };
            file.write_all(&chunk)
                .await
                .map_err(|err| DownloadError::new(format!("write {}: {err}", part.display())))?;
            hasher.update(&chunk);
            written += chunk.len() as u64;
            let done = written as f64 / entry.bytes.max(1) as f64;
            if done >= next_log {
                next_log = done + PROGRESS_STEP;
                let secs = started.elapsed().as_secs_f64().max(0.001);
                tracing::info!(
                    name: "ignis.model.download_progress",
                    file = entry.artifact_file,
                    bytes = written,
                    total_bytes = entry.bytes,
                    percent = (done * 100.0).round() as u64,
                    mib_per_sec = (((written - have) as f64 / secs) / (1024.0 * 1024.0)).round() as u64,
                    "downloading"
                );
            }
        }
        file.flush()
            .await
            .map_err(|err| DownloadError::new(format!("flush {}: {err}", part.display())))?;
        drop(file);

        // A body that stopped early is the ordinary case (a dropped
        // connection): the `.part` stays, and the next start resumes it.
        if written != entry.bytes {
            return Err(DownloadError::new(format!(
                "{}: got {written} bytes, the published artifact is {} — the partial file is kept, start again to resume it",
                entry.artifact_file, entry.bytes
            )));
        }
        let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if digest != entry.sha256 {
            // Not the model this build knows. Keeping the file would make the
            // next start resume bytes that can never verify.
            let _ = tokio::fs::remove_file(&part).await;
            return Err(DownloadError::new(format!(
                "{}: sha256 {digest} is not the published {} — the download was discarded",
                entry.artifact_file, entry.sha256
            )));
        }
        tokio::fs::rename(&part, path)
            .await
            .map_err(|err| DownloadError::new(format!("rename {}: {err}", part.display())))?;
        tracing::info!(
            name: "ignis.model.downloaded",
            model = entry.model,
            file = entry.artifact_file,
            path = %path.display(),
            bytes = entry.bytes,
            seconds = started.elapsed().as_secs(),
            "model fetched and verified"
        );
        Ok(())
    }
}

/// `<path>.part` — the name an in-flight transfer writes under, so nothing
/// ever opens a half-written artifact by its real name.
fn part_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".part");
    PathBuf::from(name)
}

/// Feed the `have` bytes already on disk through `hasher`, so a resumed
/// transfer ends with the digest of the whole file rather than of its tail.
async fn replay_into_hasher(
    part: &Path,
    have: u64,
    hasher: &mut Sha256,
) -> Result<(), DownloadError> {
    let mut file = tokio::fs::File::open(part)
        .await
        .map_err(|err| DownloadError::new(format!("open {}: {err}", part.display())))?;
    let mut buffer = vec![0u8; RESUME_CHUNK];
    let mut read_total = 0u64;
    while read_total < have {
        let want = RESUME_CHUNK.min((have - read_total) as usize);
        let read = file
            .read(&mut buffer[..want])
            .await
            .map_err(|err| DownloadError::new(format!("read {}: {err}", part.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        read_total += read as u64;
    }
    Ok(())
}

/// The `.part`, positioned to append after `have` bytes (and truncated to
/// them, so a restart never keeps the tail of an older attempt).
async fn open_part(part: &Path, have: u64) -> Result<tokio::fs::File, DownloadError> {
    let file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(part)
        .await
        .map_err(|err| DownloadError::new(format!("open {}: {err}", part.display())))?;
    file.set_len(have)
        .await
        .map_err(|err| DownloadError::new(format!("truncate {}: {err}", part.display())))?;
    let mut file = file;
    file.seek(std::io::SeekFrom::Start(have))
        .await
        .map_err(|err| DownloadError::new(format!("seek {}: {err}", part.display())))?;
    Ok(file)
}

/// Put the question to the operator on stderr and read the answer from stdin.
///
/// `None` when stdin is not a terminal — there is nobody to ask, and blocking
/// a container's start on a prompt nobody will ever answer is the one outcome
/// worse than either answer. The question goes to **stderr** because stdout
/// carries the plain lines `mk/windows/common.ps1` parses (the generated API
/// key, the public URL).
pub fn ask_on_terminal(entry: &ModelEntry, path: &Path) -> Option<bool> {
    use std::io::{BufRead, IsTerminal, Write};

    if !std::io::stdin().is_terminal() {
        return None;
    }
    let mut stderr = std::io::stderr();
    let _ = writeln!(
        stderr,
        "\nignis-server: no model at {}\n  {} is published at https://huggingface.co/{} ({:.1} GiB)\n  it will be saved as {}\n  (--no-model-download never asks; --model-download-path puts it elsewhere)",
        path.display(),
        entry.model,
        entry.repo,
        entry.gib(),
        path.display(),
    );
    let _ = write!(stderr, "Download it now? [y/N] ");
    let _ = stderr.flush();

    let mut answer = String::new();
    // EOF (a closed stdin) reads as no: the operator never said yes.
    if std::io::stdin().lock().read_line(&mut answer).ok()? == 0 {
        let _ = writeln!(stderr);
        return Some(false);
    }
    Some(matches!(
        answer.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR: &str = "models";

    fn settings<'a>(model: &'a str, dir: &'a Path) -> DownloadSettings<'a> {
        DownloadSettings {
            artifact: None,
            model,
            enabled: true,
            dir,
            supported: true,
        }
    }

    /// An `ask` that must never be reached.
    fn never_asked(_: &ModelEntry, _: &Path) -> Option<bool> {
        panic!("the operator must not be asked in this cell");
    }

    fn nothing_exists(_: &Path) -> bool {
        false
    }

    #[test]
    fn a_named_artifact_wins_whether_or_not_it_is_there() {
        // AC1: `--artifact` is the operator's word. Present or missing, the
        // decision is the same — `main` owns the failure, not the downloader.
        let dir = PathBuf::from(DIR);
        let named = PathBuf::from("/elsewhere/custom.ninfer");
        let mut settings = settings(crate::config::DEFAULT_MODEL, &dir);
        settings.artifact = Some(&named);
        assert_eq!(
            artifact_source(&settings, nothing_exists, never_asked),
            ArtifactSource::Use(named.clone())
        );
        assert_eq!(
            artifact_source(&settings, |_| true, never_asked),
            ArtifactSource::Use(named)
        );
    }

    #[test]
    fn a_build_that_cannot_run_the_model_never_downloads_it() {
        // AC1: no `cuda`, no download — `make mock` and every CPU test job
        // stay off the network by construction, not by remembering a flag.
        let dir = PathBuf::from(DIR);
        let mut settings = settings(crate::config::DEFAULT_MODEL, &dir);
        settings.supported = false;
        assert_eq!(
            artifact_source(&settings, nothing_exists, never_asked),
            ArtifactSource::Placeholder(PlaceholderReason::NotSupported)
        );
    }

    #[test]
    fn an_unknown_model_falls_through_to_the_placeholder() {
        let dir = PathBuf::from(DIR);
        let settings = settings("some-other-model", &dir);
        assert_eq!(
            artifact_source(&settings, nothing_exists, never_asked),
            ArtifactSource::Placeholder(PlaceholderReason::UnknownModel)
        );
    }

    #[test]
    fn an_artifact_already_on_disk_is_loaded_and_not_fetched_again() {
        // The everyday case on a machine that has the model: nothing is
        // asked, nothing is downloaded, and the path is the flat one.
        let dir = PathBuf::from(DIR);
        let settings = settings(crate::config::DEFAULT_MODEL, &dir);
        let expected = dir.join(REGISTRY[0].artifact_file);
        let probed = std::cell::RefCell::new(Vec::new());
        let source = artifact_source(
            &settings,
            |p| {
                probed.borrow_mut().push(p.to_path_buf());
                true
            },
            never_asked,
        );
        assert_eq!(source, ArtifactSource::Use(expected.clone()));
        assert_eq!(probed.into_inner(), vec![expected]);
    }

    #[test]
    fn downloads_off_keeps_the_placeholder_and_asks_nothing() {
        let dir = PathBuf::from(DIR);
        let mut settings = settings(crate::config::DEFAULT_MODEL, &dir);
        settings.enabled = false;
        assert_eq!(
            artifact_source(&settings, nothing_exists, never_asked),
            ArtifactSource::Placeholder(PlaceholderReason::DownloadsDisabled)
        );
    }

    #[test]
    fn downloads_off_still_loads_an_artifact_that_is_there() {
        // `--no-model-download` forbids the download, not the model: a file
        // already on disk is still what the server loads.
        let dir = PathBuf::from(DIR);
        let mut settings = settings(crate::config::DEFAULT_MODEL, &dir);
        settings.enabled = false;
        assert_eq!(
            artifact_source(&settings, |_| true, never_asked),
            ArtifactSource::Use(dir.join(REGISTRY[0].artifact_file))
        );
    }

    #[test]
    fn a_tty_is_asked_and_yes_downloads() {
        let dir = PathBuf::from(DIR);
        let settings = settings(crate::config::DEFAULT_MODEL, &dir);
        let source = artifact_source(&settings, nothing_exists, |entry, path| {
            // The question knows what it is about to spend, and where.
            assert_eq!(entry.model, crate::config::DEFAULT_MODEL);
            assert_eq!(path, dir.join(entry.artifact_file));
            Some(true)
        });
        assert_eq!(
            source,
            ArtifactSource::Download {
                entry: &REGISTRY[0],
                dir: dir.clone()
            }
        );
    }

    #[test]
    fn a_tty_that_says_no_keeps_the_placeholder() {
        let dir = PathBuf::from(DIR);
        let settings = settings(crate::config::DEFAULT_MODEL, &dir);
        assert_eq!(
            artifact_source(&settings, nothing_exists, |_, _| Some(false)),
            ArtifactSource::Placeholder(PlaceholderReason::Declined)
        );
    }

    #[test]
    fn nobody_to_ask_downloads() {
        // A container or a daemon: no TTY, so no question — and coming up on
        // the placeholder would be the wrong surprise.
        let dir = PathBuf::from(DIR);
        let settings = settings(crate::config::DEFAULT_MODEL, &dir);
        assert_eq!(
            artifact_source(&settings, nothing_exists, |_, _| None),
            ArtifactSource::Download {
                entry: &REGISTRY[0],
                dir
            }
        );
    }

    #[test]
    fn the_registry_key_is_the_model_id_case_insensitively() {
        assert_eq!(
            registry_entry("QWEN3.8-27B").map(|e| e.repo),
            Some(REGISTRY[0].repo)
        );
        assert!(registry_entry("qwen3.8-27b-instruct").is_none());
    }

    #[test]
    fn every_registry_entry_names_its_own_sidecar_and_a_pinned_digest() {
        // The loader needs the sidecar next to the artifact under one of
        // `loader::SIDECAR_SUFFIXES`, and the transfer needs 64 hex digits to
        // check what it wrote against.
        for entry in REGISTRY {
            assert!(
                crate::loader::SIDECAR_SUFFIXES
                    .iter()
                    .any(|suffix| entry.sidecar_file == format!("{}{suffix}", entry.artifact_file)),
                "{}: the sidecar must be the artifact's name plus a recognized suffix",
                entry.model
            );
            assert_eq!(entry.sha256.len(), 64, "{}", entry.model);
            assert!(
                entry
                    .sha256
                    .chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{}: the digest is lowercase hex",
                entry.model
            );
            assert!(entry.bytes > 0, "{}", entry.model);
        }
    }
}
