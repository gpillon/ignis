//! The transfer (spec model-download/01 AC3-AC5, spec model-download/02
//! §The transfer): every file of a catalog entry, sidecars and companion
//! included, through one verified path.
//!
//! A file streams to `<name>.part`, is hashed as it is written, and is
//! renamed into place only once its byte count and digest are the pinned
//! ones. What becomes of the `.part` says which failure it was: a body that
//! stopped early keeps it, because that is a dropped connection and the next
//! attempt resumes it with `Range`; a body that hashes wrong, one that runs
//! past the pin, and a partial the source answers `416` to are deleted —
//! they are not this file, and keeping them would make every later attempt
//! resume bytes that can never verify.
//!
//! The endpoint is injectable, so the whole path is covered by tests against
//! loopback servers (`tests/model_download.rs`), never the internet.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use super::catalog::{CatalogEntry, CatalogFile};
use crate::config::ApiKey;

/// How long the connection alone may take. The transfer as a whole is
/// deliberately **not** capped: 72 GB over a slow line is a long download,
/// not a hung one.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the transfer may go without a single byte arriving. A slow line
/// keeps its bytes coming and is never cut; a peer that has stopped talking
/// altogether must not hold the start open forever, since nothing downstream
/// of it has a deadline of its own.
const READ_TIMEOUT: Duration = Duration::from_secs(120);

/// Read in this much of the `.part` at a time when a resumed transfer replays
/// it through the hasher.
const RESUME_CHUNK: usize = 1 << 20;

/// How much of a file must arrive between two progress reports.
const PROGRESS_STEP: f64 = 0.05;

/// A transfer that did not produce a verified file. Never a panic, and never
/// a partially-written file the next attempt would trust.
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

/// Where a transfer says what it is doing.
#[derive(Clone)]
pub enum Report {
    /// Structured log events (`ignis.model.*`): a server start, whose log
    /// is the record of what it did.
    Log,
    /// One plain line per event to this sink: `model download`, whose stdout
    /// carries only the paths it fetched, so its progress goes to stderr.
    Lines(Arc<dyn Fn(&str) + Send + Sync>),
}

/// What a transfer reports, whichever way it reports it.
enum Event<'a> {
    /// A file already at its final name with its pinned byte count.
    Present { file: &'a CatalogFile, path: &'a Path },
    /// A file's request went out.
    Started { entry: &'a CatalogEntry, file: &'a CatalogFile, url: &'a str, path: &'a Path, resume_from: u64 },
    /// The source answered a ranged request with the whole body.
    ResumeUnsupported { file: &'a CatalogFile, have: u64 },
    /// Another [`PROGRESS_STEP`] of a file arrived.
    Progress { file: &'a CatalogFile, bytes: u64, mib_per_sec: u64 },
    /// A file verified and was renamed into place.
    Fetched { entry: &'a CatalogEntry, file: &'a CatalogFile, path: &'a Path, seconds: u64 },
}

/// Fetches a [`CatalogEntry`]'s files into a directory, from one endpoint.
pub struct Downloader {
    client: reqwest::Client,
    endpoint: String,
    /// Sent as `Authorization: Bearer …` to the endpoint, and never across a
    /// redirect to another host: `reqwest` drops the header itself when a
    /// redirect leaves the host or the port (`tests/model_download.rs` holds
    /// it to that) — Hugging Face sends a large file's body from its CDN.
    token: Option<ApiKey>,
    report: Report,
}

impl Downloader {
    /// One fetching from `endpoint` (`https://host[/prefix]`, a trailing
    /// slash or none) with `token` ([`super::bearer_token`]'s choice),
    /// reporting to the log.
    pub fn new(endpoint: &str, token: Option<ApiKey>) -> Result<Self, DownloadError> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .user_agent(concat!("ignis/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|err| DownloadError::new(format!("HTTP client: {err}")))?;
        let endpoint = endpoint.trim_end_matches('/').to_owned();
        Ok(Self { client, endpoint, token, report: Report::Log })
    }

    /// The same, reporting as `report` says.
    pub fn reporting(self, report: Report) -> Self {
        Self { report, ..self }
    }

    /// The URL a file of `entry` resolves to: Hugging Face's own route,
    /// `{endpoint}/{repo}/resolve/{revision}/{file}`, which every mirror the
    /// owner named also answers. The revision is percent-encoded (a branch
    /// may hold a `/`), and so is every other segment — harmless for the
    /// plain names the catalog allows.
    pub fn url(&self, entry: &CatalogEntry, file: &str) -> String {
        let repo: Vec<String> = entry.repo.split('/').map(percent_encode).collect();
        format!("{}/{}/resolve/{}/{}", self.endpoint, repo.join("/"), percent_encode(&entry.revision), percent_encode(file))
    }

    /// Fetch every file of `entry` into `dir` and return the artifact's path.
    ///
    /// A file already at its final name with its pinned byte count is kept
    /// without hashing it; one there with another byte count is refused by
    /// name and left alone — checked for every file before any byte is
    /// fetched. The rest are fetched in ascending byte order, so a repo that
    /// cannot serve its small files fails in a second instead of an hour.
    pub async fn fetch(&self, entry: &CatalogEntry, dir: &Path) -> Result<PathBuf, DownloadError> {
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(|err| DownloadError::new(format!("cannot create {}: {err}", dir.display())))?;
        let mut missing = Vec::new();
        for file in entry.files_by_size() {
            let path = dir.join(&file.name);
            match tokio::fs::metadata(&path).await {
                Ok(meta) if meta.is_file() && meta.len() == file.bytes => self.say(Event::Present { file, path: &path }),
                Ok(meta) => {
                    return Err(DownloadError::new(format!(
                        "{}: {} is already there with {} bytes, not the published {} — it was left alone; move it aside to fetch the published file",
                        file.name,
                        path.display(),
                        meta.len(),
                        file.bytes
                    )));
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => missing.push((file, path)),
                Err(err) => return Err(DownloadError::new(format!("{}: {err}", path.display()))),
            }
        }
        for (file, path) in missing {
            self.fetch_file(entry, file, &path).await?;
        }
        Ok(entry.artifact_path(dir))
    }

    /// One file. Resumes an earlier `.part` when there is one, verifies
    /// length and digest, and only then renames.
    async fn fetch_file(&self, entry: &CatalogEntry, file: &CatalogFile, path: &Path) -> Result<(), DownloadError> {
        let part = part_path(path);
        let url = self.url(entry, &file.name);
        // What an earlier attempt left. As many bytes as the pin names (or
        // more) is not a resume point but a different file — the digest check
        // never passed on it, so it is discarded rather than continued.
        let mut have = match tokio::fs::metadata(&part).await {
            Ok(meta) if meta.len() < file.bytes => meta.len(),
            Ok(_) => {
                let _ = tokio::fs::remove_file(&part).await;
                0
            }
            Err(_) => 0,
        };
        let mut hasher = Sha256::new();
        if have > 0 {
            replay_into_hasher(&part, have, &mut hasher).await?;
        }

        let mut request = self.client.get(&url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token.as_str());
        }
        if have > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut response = request
            .send()
            .await
            .map_err(|err| DownloadError::new(format!("GET {url}: {err}")))?;
        let status = response.status();
        // The source has no bytes past where the partial file ends: it is not
        // the file this `.part` was the start of. Keeping it would send the
        // same unsatisfiable range on every attempt from now on, so it goes.
        if have > 0 && status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            let _ = tokio::fs::remove_file(&part).await;
            return Err(DownloadError::new(format!(
                "{}: the source has no bytes past {have}, where the partial download ends — it was discarded, start again to fetch the file whole",
                file.name
            )));
        }
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
            self.say(Event::ResumeUnsupported { file, have });
            have = 0;
            hasher = Sha256::new();
        }

        let mut out = open_part(&part, have).await?;
        self.say(Event::Started { entry, file, url: &url, path, resume_from: have });

        let started = Instant::now();
        let mut written = have;
        let mut next_report = written as f64 / file.bytes.max(1) as f64 + PROGRESS_STEP;
        loop {
            let chunk = response
                .chunk()
                .await
                .map_err(|err| DownloadError::new(format!("GET {url}: {err}")))?;
            let Some(chunk) = chunk else { break };
            out.write_all(&chunk)
                .await
                .map_err(|err| DownloadError::new(format!("write {}: {err}", part.display())))?;
            hasher.update(&chunk);
            written += chunk.len() as u64;
            // A body longer than the pin is a different file, whatever its
            // first bytes were. Stopping here costs one chunk instead of the
            // rest of the transfer, and the part file goes with it.
            if written > file.bytes {
                drop(out);
                let _ = tokio::fs::remove_file(&part).await;
                return Err(DownloadError::new(format!(
                    "{}: the source is longer than the published {} bytes — it is not this file, and the download was discarded",
                    file.name, file.bytes
                )));
            }
            let done = written as f64 / file.bytes.max(1) as f64;
            if done >= next_report {
                next_report = done + PROGRESS_STEP;
                let secs = started.elapsed().as_secs_f64().max(0.001);
                let mib_per_sec = (((written - have) as f64 / secs) / (1024.0 * 1024.0)).round() as u64;
                self.say(Event::Progress { file, bytes: written, mib_per_sec });
            }
        }
        out.flush()
            .await
            .map_err(|err| DownloadError::new(format!("flush {}: {err}", part.display())))?;
        drop(out);

        // A body that stopped early is the ordinary case (a dropped
        // connection): the `.part` stays, and the next attempt resumes it.
        if written != file.bytes {
            return Err(DownloadError::new(format!(
                "{}: got {written} bytes, the published file is {} — the partial file is kept, start again to resume it",
                file.name, file.bytes
            )));
        }
        let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if digest != file.sha256 {
            // Not the file this catalog pins. Keeping it would make the next
            // attempt resume bytes that can never verify.
            let _ = tokio::fs::remove_file(&part).await;
            return Err(DownloadError::new(format!(
                "{}: sha256 {digest} is not the published {} — the download was discarded",
                file.name, file.sha256
            )));
        }
        tokio::fs::rename(&part, path)
            .await
            .map_err(|err| DownloadError::new(format!("rename {}: {err}", part.display())))?;
        self.say(Event::Fetched { entry, file, path, seconds: started.elapsed().as_secs() });
        Ok(())
    }

    /// Report `event` the way this downloader reports. No event carries the
    /// token: a URL is the endpoint, the repo, the revision and the file.
    fn say(&self, event: Event<'_>) {
        let lines = match &self.report {
            Report::Log => return log(event),
            Report::Lines(lines) => lines,
        };
        let line = match event {
            Event::Present { file, path } => format!("{}: already at {} with its {} bytes, kept", file.name, path.display(), file.bytes),
            Event::Started { file, url, resume_from, .. } if resume_from > 0 => {
                format!("{}: resuming at byte {resume_from} of {} from {url}", file.name, file.bytes)
            }
            Event::Started { file, url, .. } => format!("{}: fetching {} bytes from {url}", file.name, file.bytes),
            Event::ResumeUnsupported { file, have } => {
                format!("{}: the source ignored the range from byte {have}, starting the file over", file.name)
            }
            Event::Progress { file, bytes, mib_per_sec } => format!(
                "{}: {}% ({bytes} of {} bytes, {mib_per_sec} MiB/s)",
                file.name,
                (bytes as f64 * 100.0 / file.bytes.max(1) as f64).round() as u64,
                file.bytes
            ),
            Event::Fetched { file, path, .. } => format!("{}: verified, saved as {}", file.name, path.display()),
        };
        lines(&line);
    }
}

/// [`Report::Log`]: the `ignis.model.*` events a start logs.
fn log(event: Event<'_>) {
    match event {
        Event::Present { file, path } => tracing::info!(
            name: "ignis.model.file_present",
            file = %file.name,
            path = %path.display(),
            bytes = file.bytes,
            "already on disk at its published size, kept"
        ),
        Event::Started { entry, file, url, path, resume_from } => tracing::info!(
            name: "ignis.model.download_started",
            model = %entry.id,
            repo = %entry.repo,
            revision = %entry.revision,
            file = %file.name,
            url = %url,
            path = %path.display(),
            total_bytes = file.bytes,
            resume_from,
            "fetching the model"
        ),
        Event::ResumeUnsupported { file, have } => tracing::warn!(
            name: "ignis.model.resume_unsupported",
            file = %file.name,
            have,
            "the source ignored Range — starting the transfer over"
        ),
        Event::Progress { file, bytes, mib_per_sec } => tracing::info!(
            name: "ignis.model.download_progress",
            file = %file.name,
            bytes,
            total_bytes = file.bytes,
            percent = (bytes as f64 * 100.0 / file.bytes.max(1) as f64).round() as u64,
            mib_per_sec,
            "downloading"
        ),
        Event::Fetched { entry, file, path, seconds } => tracing::info!(
            name: "ignis.model.downloaded",
            model = %entry.id,
            file = %file.name,
            path = %path.display(),
            bytes = file.bytes,
            seconds,
            "fetched and verified"
        ),
    }
}

/// RFC 3986 percent-encoding of one path segment: every byte but the
/// unreserved ones (`A-Z a-z 0-9 - . _ ~`) as `%XX`.
pub fn percent_encode(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// `<path>.part` — the name an in-flight transfer writes under, so nothing
/// ever opens a half-written file by its real name.
pub(crate) fn part_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".part");
    PathBuf::from(name)
}

/// Feed the `have` bytes already on disk through `hasher`, so a resumed
/// transfer ends with the digest of the whole file rather than of its tail.
async fn replay_into_hasher(part: &Path, have: u64, hasher: &mut Sha256) -> Result<(), DownloadError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_segment_keeps_the_unreserved_characters_and_encodes_the_rest() {
        assert_eq!(percent_encode("e961b419b672e183aa55df8c4b975abc82006e8a"), "e961b419b672e183aa55df8c4b975abc82006e8a");
        assert_eq!(percent_encode("refs/pr/7"), "refs%2Fpr%2F7");
        assert_eq!(percent_encode("v1.0_rc-2~x"), "v1.0_rc-2~x");
        assert_eq!(percent_encode("a b%é"), "a%20b%25%C3%A9");
    }

    #[test]
    fn the_part_file_is_the_final_name_plus_part() {
        assert_eq!(part_path(Path::new("models/a.ninfer")), PathBuf::from("models/a.ninfer.part"));
    }
}
