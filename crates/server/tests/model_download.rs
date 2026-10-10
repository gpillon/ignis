//! The model transfer (GitHub #234 and #313; spec model-download/01 AC3-AC5,
//! spec model-download/02 AC 4, 6-9 and 12), driven against local `axum`
//! servers: no network, no GPU, no 19 GB.
//!
//! What is pinned here is what protects the operator from a file that is not
//! the model: every file of an entry — sidecars included — is checked
//! against its byte count and digest before it is renamed into place, the
//! small files are fetched first, an interrupted transfer resumes from its
//! own `.part`, and the token goes to the endpoint and nowhere else.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Path as UrlPath, State};
use axum::http::{header, HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;

use ignis_server::config::file::Files;
use ignis_server::config::{resolve_with, ApiKey, ConfigOutcome};
use ignis_server::download::command::run;
use ignis_server::download::{CatalogEntry, CatalogFile, CatalogLayer, Downloader, ModelCommand, Report};

// ── the repo this test serves ───────────────────────────────────────────────

const ARTIFACT: &str = "test-model.ninfer";
const SIDECAR: &str = "test-model.ninfer.graft.json";
const COMPANION: &str = "test-model-mtp.ninfer";
const REPO: &str = "ignis-test/model";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const SIDECAR_BODY: &[u8] = br#"{"recipe_id":"test"}"#;

/// The bytes the fake repo serves as the artifact: long enough that a resume
/// has something to resume from.
fn artifact_bytes() -> Vec<u8> {
    (0..64 * 1024u32).map(|i| (i % 251) as u8).collect()
}

/// A companion container between the sidecar and the artifact in size.
fn companion_bytes() -> Vec<u8> {
    (0..8 * 1024u32).map(|i| (i % 13) as u8).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn pinned(name: &str, body: &[u8]) -> CatalogFile {
    CatalogFile { name: name.to_owned(), bytes: body.len() as u64, sha256: sha256_hex(body) }
}

/// The catalog entry for the served repo: sidecar and artifact, pinned to
/// what the repo serves. A test tampers with a pin to make a check fail.
fn entry() -> CatalogEntry {
    CatalogEntry {
        id: "test-model".to_owned(),
        repo: REPO.to_owned(),
        revision: REVISION.to_owned(),
        artifact: ARTIFACT.to_owned(),
        // Listed large first: the fetch order is the transfer's, not the list's.
        files: vec![pinned(ARTIFACT, &artifact_bytes()), pinned(SIDECAR, SIDECAR_BODY)],
        layer: CatalogLayer::Operator,
    }
}

/// [`entry`] with the companion container too: three files to order.
fn three_file_entry() -> CatalogEntry {
    let mut entry = entry();
    entry.files.insert(1, pinned(COMPANION, &companion_bytes()));
    entry
}

// ── the fake Hugging Face ───────────────────────────────────────────────────

/// One request the repo route was sent.
#[derive(Debug, Clone)]
struct Seen {
    file: String,
    range: Option<String>,
    authorization: Option<String>,
}

#[derive(Default)]
struct Served {
    /// Every request the repo route was sent, in order.
    seen: Mutex<Vec<Seen>>,
    /// The raw request path of each, as it arrived (still percent-encoded).
    raw_paths: Mutex<Vec<String>>,
    /// The revision each was for, decoded.
    revisions: Mutex<Vec<String>>,
    /// Body bytes actually written into responses, per file.
    bytes_sent: Mutex<BTreeMap<String, usize>>,
    /// Cut every artifact response off after this many bytes (0 = serve all).
    truncate_after: AtomicUsize,
    /// Ignore `Range` and always answer `200` with the whole body.
    ignore_range: AtomicUsize,
    /// Answer every ranged request `416`, as a source whose file is shorter
    /// than the range asked for does.
    range_not_satisfiable: AtomicUsize,
    /// Serve this many extra bytes past the artifact (0 = none).
    extra_bytes: AtomicUsize,
    /// Answer every repo request with a redirect to this base instead (the
    /// CDN Hugging Face sends a large file from).
    redirect_to: Mutex<Option<String>>,
}

impl Served {
    fn sent(&self, file: &str) -> usize {
        self.bytes_sent.lock().unwrap().get(file).copied().unwrap_or(0)
    }

    fn requested(&self) -> Vec<String> {
        self.seen.lock().unwrap().iter().map(|seen| seen.file.clone()).collect()
    }
}

/// What the repo holds at [`REVISION`].
fn body_of(file: &str) -> Option<Vec<u8>> {
    match file {
        ARTIFACT => Some(artifact_bytes()),
        SIDECAR => Some(SIDECAR_BODY.to_vec()),
        COMPANION => Some(companion_bytes()),
        _ => None,
    }
}

/// `GET [/prefix]/{owner}/{name}/resolve/{revision}/{file}` — the route
/// Hugging Face serves a repo file under, with `Range` support.
async fn resolve(
    State(state): State<Arc<Served>>,
    UrlPath((owner, name, revision, file)): UrlPath<(String, String, String, String)>,
    uri: Uri,
    headers: HeaderMap,
) -> Response {
    let header_text = |name| headers.get(name).and_then(|v: &header::HeaderValue| v.to_str().ok()).map(str::to_owned);
    let range = header_text(header::RANGE);
    state.seen.lock().unwrap().push(Seen { file: file.clone(), range: range.clone(), authorization: header_text(header::AUTHORIZATION) });
    state.raw_paths.lock().unwrap().push(uri.path().to_owned());
    state.revisions.lock().unwrap().push(revision.clone());
    if format!("{owner}/{name}") != REPO {
        return StatusCode::NOT_FOUND.into_response();
    }
    if let Some(base) = state.redirect_to.lock().unwrap().clone() {
        return (StatusCode::FOUND, [(header::LOCATION, format!("{base}/cdn/{file}"))]).into_response();
    }
    serve(&state, &file, range)
}

/// The CDN's route: the same bodies, no repo, no revision.
async fn cdn(State(state): State<Arc<Served>>, UrlPath(file): UrlPath<String>, headers: HeaderMap) -> Response {
    let header_text = |name| headers.get(name).and_then(|v: &header::HeaderValue| v.to_str().ok()).map(str::to_owned);
    let range = header_text(header::RANGE);
    state.seen.lock().unwrap().push(Seen { file: file.clone(), range: range.clone(), authorization: header_text(header::AUTHORIZATION) });
    serve(&state, &file, range)
}

fn serve(state: &Served, file: &str, range: Option<String>) -> Response {
    let Some(mut body) = body_of(file) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let is_artifact = file == ARTIFACT;
    if range.is_some() && state.range_not_satisfiable.load(Ordering::SeqCst) == 1 {
        return StatusCode::RANGE_NOT_SATISFIABLE.into_response();
    }
    if is_artifact {
        body.extend(std::iter::repeat_n(0xABu8, state.extra_bytes.load(Ordering::SeqCst)));
    }
    let ignore_range = state.ignore_range.load(Ordering::SeqCst) == 1;
    let start = match (&range, ignore_range) {
        (Some(raw), false) => raw
            .trim_start_matches("bytes=")
            .trim_end_matches('-')
            .parse::<usize>()
            .unwrap_or(0),
        _ => 0,
    };
    let mut served = body[start.min(body.len())..].to_vec();
    let truncate_after = state.truncate_after.load(Ordering::SeqCst);
    if is_artifact && truncate_after > 0 {
        served.truncate(truncate_after);
    }
    *state.bytes_sent.lock().unwrap().entry(file.to_owned()).or_default() += served.len();
    let status = if start > 0 && !ignore_range {
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    (status, Body::from(served)).into_response()
}

/// A fake repo on a loopback port, under `prefix` (`""` for Hugging Face's
/// own layout, `/mirror/hf` for a proxy's), and the state a test steers it
/// with.
async fn repo_server_at(prefix: &str) -> (SocketAddr, Arc<Served>) {
    let state = Arc::new(Served::default());
    let app = Router::new()
        .route(&format!("{prefix}/{{owner}}/{{name}}/resolve/{{revision}}/{{file}}"), get(resolve))
        .route("/cdn/{file}", get(cdn))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (addr, state)
}

async fn repo_server() -> (SocketAddr, Arc<Served>) {
    repo_server_at("").await
}

fn downloader(addr: SocketAddr) -> Downloader {
    Downloader::new(&format!("http://{addr}"), None).expect("downloader")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-download-{tag}-{}-{}", std::process::id(), rand_suffix()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A per-call suffix, so two tests in the same process never share a
/// directory (no `rand` dependency for a few digits).
fn rand_suffix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").subsec_nanos() as u64;
    nanos * 1_000 + CALLS.fetch_add(1, Ordering::SeqCst) as u64
}

fn part(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.part"))
}

// ── the transfer ────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_clean_transfer_writes_every_file_of_the_entry() {
    // 01 AC3/AC5: the sidecar lands next to the artifact under the name the
    // loader looks for, every file matches its pin, and no `.part` survives.
    let (addr, state) = repo_server().await;
    let dir = temp_dir("clean");

    let path = downloader(addr).fetch(&entry(), &dir).await.expect("fetch");

    assert_eq!(path, dir.join(ARTIFACT));
    assert_eq!(std::fs::read(&path).expect("artifact"), artifact_bytes());
    assert_eq!(std::fs::read(dir.join(SIDECAR)).expect("sidecar"), SIDECAR_BODY);
    assert!(!part(&dir, ARTIFACT).exists() && !part(&dir, SIDECAR).exists(), "the part files are renamed, not left behind");
    assert_eq!(state.requested(), [SIDECAR, ARTIFACT], "the small file first");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_body_that_does_not_match_the_pinned_digest_is_refused() {
    // 01 AC3: the bytes arrived intact, but they are not the model this
    // catalog pins — nothing is renamed into place, and the error says so.
    let (addr, _state) = repo_server().await;
    let mut entry = entry();
    entry.files[0].sha256 = "0".repeat(64);
    let dir = temp_dir("tampered");

    let err = downloader(addr).fetch(&entry, &dir).await.expect_err("a wrong digest must refuse the transfer");

    assert!(err.to_string().contains("sha256") && err.to_string().contains(ARTIFACT), "{err}");
    assert!(!dir.join(ARTIFACT).exists(), "no artifact is left in place");
    assert!(!part(&dir, ARTIFACT).exists(), "the rejected part file is removed, so the next attempt does not resume it");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_sidecar_served_with_the_wrong_digest_is_discarded_and_the_fetch_fails() {
    // 02 AC 8: a sidecar is no longer fetched unverified — one that is not
    // the pinned record goes, the fetch fails, and the large file is never
    // asked for.
    let (addr, state) = repo_server().await;
    let mut entry = entry();
    entry.files[1].sha256 = "f".repeat(64);
    let dir = temp_dir("bad-sidecar");

    let err = downloader(addr).fetch(&entry, &dir).await.expect_err("an unverified sidecar must fail the fetch");

    assert!(err.to_string().contains(SIDECAR) && err.to_string().contains("sha256"), "{err}");
    assert!(!dir.join(SIDECAR).exists() && !part(&dir, SIDECAR).exists(), "nothing of it is kept");
    assert_eq!(state.requested(), [SIDECAR]);
    assert_eq!(state.sent(ARTIFACT), 0, "not one byte of the body is fetched");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_short_body_is_refused_before_the_digest_is_even_checked() {
    // 01 AC3: a connection that dies mid-transfer leaves fewer bytes than the
    // pin names. The `.part` stays, so the next attempt resumes it.
    let (addr, state) = repo_server().await;
    let dir = temp_dir("short");
    state.truncate_after.store(4096, Ordering::SeqCst);

    let err = downloader(addr).fetch(&entry(), &dir).await.expect_err("a short body must refuse the transfer");

    assert!(err.to_string().contains("4096"), "{err}");
    assert!(!dir.join(ARTIFACT).exists());
    assert_eq!(std::fs::metadata(part(&dir, ARTIFACT)).expect("the part file survives an interrupted transfer").len(), 4096);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn an_interrupted_transfer_resumes_from_the_part_file() {
    // 01 AC4: the second run asks for the rest with `Range`, hashes the bytes
    // already on disk, and ends with the same verified file — without
    // fetching the first 4 KiB twice, nor the sidecar it already has.
    let (addr, state) = repo_server().await;
    let body = artifact_bytes();
    let dir = temp_dir("resume");
    let downloader = downloader(addr);

    state.truncate_after.store(4096, Ordering::SeqCst);
    downloader.fetch(&entry(), &dir).await.expect_err("the first attempt is cut short");
    state.truncate_after.store(0, Ordering::SeqCst);
    let path = downloader.fetch(&entry(), &dir).await.expect("resumed fetch");

    assert_eq!(std::fs::read(&path).expect("artifact"), body);
    let artifact_ranges: Vec<Option<String>> =
        state.seen.lock().unwrap().iter().filter(|seen| seen.file == ARTIFACT).map(|seen| seen.range.clone()).collect();
    assert_eq!(artifact_ranges, [None, Some("bytes=4096-".to_owned())], "the whole file, then only what is missing");
    assert_eq!(state.requested(), [SIDECAR, ARTIFACT, ARTIFACT], "the verified sidecar is not fetched again");
    assert_eq!(state.sent(ARTIFACT), body.len(), "the resumed bytes are transferred once each");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_server_that_ignores_the_range_restarts_the_transfer_cleanly() {
    // 01 AC4, the other branch: answering `200` to a `Range` request means
    // the body starts at zero. Appending it to the part file would corrupt
    // it, so the transfer starts over — and still verifies.
    let (addr, state) = repo_server().await;
    let dir = temp_dir("norange");
    let downloader = downloader(addr);

    state.truncate_after.store(4096, Ordering::SeqCst);
    downloader.fetch(&entry(), &dir).await.expect_err("cut short");
    state.truncate_after.store(0, Ordering::SeqCst);
    state.ignore_range.store(1, Ordering::SeqCst);
    let path = downloader.fetch(&entry(), &dir).await.expect("restarted fetch");

    assert_eq!(std::fs::read(&path).expect("artifact"), artifact_bytes());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_source_longer_than_the_pin_is_discarded_mid_transfer() {
    // A body still arriving past the published size is a different file,
    // whatever its first bytes were: stop at the chunk that crosses the line
    // rather than write the rest, and leave nothing to resume.
    let (addr, state) = repo_server().await;
    let dir = temp_dir("toolong");
    state.extra_bytes.store(8192, Ordering::SeqCst);

    let err = downloader(addr).fetch(&entry(), &dir).await.expect_err("a longer source must refuse the transfer");

    assert!(err.to_string().contains("longer"), "{err}");
    assert!(!dir.join(ARTIFACT).exists());
    assert!(!part(&dir, ARTIFACT).exists(), "nothing is left for the next attempt to resume");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_partial_the_source_cannot_satisfy_is_discarded_instead_of_retried_forever() {
    // The source's file ends before the `.part` does (it was republished
    // shorter): the ranged request comes back `416`. Keeping the partial file
    // would send the same unsatisfiable range on every attempt from now on.
    let (addr, state) = repo_server().await;
    let dir = temp_dir("unsatisfiable");
    let downloader = downloader(addr);

    state.truncate_after.store(4096, Ordering::SeqCst);
    downloader.fetch(&entry(), &dir).await.expect_err("cut short");
    state.truncate_after.store(0, Ordering::SeqCst);
    state.range_not_satisfiable.store(1, Ordering::SeqCst);
    let err = downloader.fetch(&entry(), &dir).await.expect_err("416 must refuse the transfer");
    assert!(err.to_string().contains("4096"), "{err}");
    assert!(!part(&dir, ARTIFACT).exists(), "the unresumable part file is discarded");

    // And the attempt after that one fetches the artifact whole.
    state.range_not_satisfiable.store(0, Ordering::SeqCst);
    let path = downloader.fetch(&entry(), &dir).await.expect("clean fetch");
    assert_eq!(std::fs::read(&path).expect("artifact"), artifact_bytes());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_repo_without_the_sidecar_fails_before_the_artifact_is_fetched() {
    // 01 AC5: the sidecar is a second-long request and the loader refuses a
    // load without it — so it is fetched first, and its failure costs
    // nothing.
    let (addr, state) = repo_server().await;
    let mut entry = entry();
    entry.files[1].name = "test-model.ninfer.absent.json".to_owned();
    let dir = temp_dir("nosidecar");

    let err = downloader(addr).fetch(&entry, &dir).await.expect_err("a missing sidecar must fail the transfer");

    assert!(err.to_string().contains("404"), "{err}");
    assert_eq!(state.sent(ARTIFACT), 0, "not one byte of the 19 GB body is fetched");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_multi_file_entry_goes_small_first_each_file_resuming_from_its_own_part() {
    // 02 AC 9: whatever order the catalog lists them in, the files are
    // fetched in ascending byte order; a file with a `.part` of its own
    // resumes from it; one already complete at its final name is kept
    // without being fetched (or hashed).
    let (addr, state) = repo_server().await;
    let dir = temp_dir("multi");
    std::fs::create_dir_all(&dir).unwrap();
    let entry = three_file_entry();
    assert_eq!(entry.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), [ARTIFACT, COMPANION, SIDECAR]);
    // The companion's first 1000 bytes, as an interrupted earlier fetch
    // left them.
    std::fs::write(part(&dir, COMPANION), &companion_bytes()[..1000]).unwrap();

    downloader(addr).fetch(&entry, &dir).await.expect("fetch");
    assert_eq!(state.requested(), [SIDECAR, COMPANION, ARTIFACT]);
    let companion_range = state.seen.lock().unwrap().iter().find(|seen| seen.file == COMPANION).and_then(|seen| seen.range.clone());
    assert_eq!(companion_range.as_deref(), Some("bytes=1000-"));
    assert_eq!(std::fs::read(dir.join(COMPANION)).unwrap(), companion_bytes());

    // Fetched again: every file is complete at its final name, so nothing is
    // requested. The size alone decides — the bytes are not hashed.
    std::fs::write(dir.join(SIDECAR), vec![b'x'; SIDECAR_BODY.len()]).unwrap();
    downloader(addr).fetch(&entry, &dir).await.expect("nothing to fetch");
    assert_eq!(state.requested().len(), 3, "no request for a complete file");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_file_at_its_final_name_with_another_size_is_refused_and_left_in_place() {
    // 02 AC 9: a final name holding another byte count is not this file and
    // not a resume point either: refused by name before any byte is
    // fetched, and left exactly as it was.
    let (addr, state) = repo_server().await;
    let dir = temp_dir("wrong-size");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(ARTIFACT), b"someone else's file").unwrap();

    let err = downloader(addr).fetch(&entry(), &dir).await.expect_err("a wrong-size file must refuse the fetch");

    let message = err.to_string();
    assert!(message.contains(ARTIFACT) && message.contains("left alone"), "{message}");
    assert_eq!(std::fs::read(dir.join(ARTIFACT)).unwrap(), b"someone else's file");
    assert!(state.requested().is_empty(), "checked before anything is fetched: {:?}", state.requested());
    let _ = std::fs::remove_dir_all(&dir);
}

// ── the endpoint and the token ──────────────────────────────────────────────

#[tokio::test]
async fn a_files_url_carries_the_endpoint_the_repo_the_revision_and_the_name() {
    // 02 AC 4: `{endpoint}/{repo}/resolve/{revision}/{file}`, an endpoint
    // with a path of its own (a proxy's) kept, the revision percent-encoded.
    let (addr, state) = repo_server_at("/mirror/hf").await;
    let mut entry = entry();
    entry.revision = "refs/pr/7".to_owned();
    let dir = temp_dir("url");
    let downloader = Downloader::new(&format!("http://{addr}/mirror/hf/"), None).expect("downloader");

    assert_eq!(
        downloader.url(&entry, SIDECAR),
        format!("http://{addr}/mirror/hf/ignis-test/model/resolve/refs%2Fpr%2F7/{SIDECAR}")
    );
    downloader.fetch(&entry, &dir).await.expect("fetched through the mirror's prefix");
    assert_eq!(
        *state.raw_paths.lock().unwrap(),
        [format!("/mirror/hf/ignis-test/model/resolve/refs%2Fpr%2F7/{SIDECAR}"), format!("/mirror/hf/ignis-test/model/resolve/refs%2Fpr%2F7/{ARTIFACT}")]
    );
    assert_eq!(*state.revisions.lock().unwrap(), ["refs/pr/7", "refs/pr/7"], "one segment, decoded back to the branch");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn the_token_goes_to_the_endpoint_and_never_across_a_redirect_to_another_host() {
    // 02 AC 6: two listeners. The endpoint (`127.0.0.1`) answers every file
    // with a redirect to the other under another host name (`localhost`, as
    // Hugging Face sends a large body from its CDN): the endpoint sees the
    // token, the other never does — and a resume's `Range` still reaches it.
    let (endpoint, at_endpoint) = repo_server().await;
    let (cdn, at_cdn) = repo_server().await;
    *at_endpoint.redirect_to.lock().unwrap() = Some(format!("http://localhost:{}", cdn.port()));
    let dir = temp_dir("redirect");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(part(&dir, ARTIFACT), &artifact_bytes()[..512]).unwrap();
    let downloader = Downloader::new(&format!("http://{endpoint}"), Some(ApiKey::new("tok-secret"))).expect("downloader");

    let path = downloader.fetch(&entry(), &dir).await.expect("fetched through the redirect");

    assert_eq!(std::fs::read(path).unwrap(), artifact_bytes());
    let to_endpoint = at_endpoint.seen.lock().unwrap().clone();
    let to_cdn = at_cdn.seen.lock().unwrap().clone();
    assert_eq!(to_endpoint.len(), 2, "{to_endpoint:?}");
    assert!(to_endpoint.iter().all(|seen| seen.authorization.as_deref() == Some("Bearer tok-secret")), "{to_endpoint:?}");
    assert_eq!(to_cdn.iter().map(|seen| seen.file.as_str()).collect::<Vec<_>>(), [SIDECAR, ARTIFACT]);
    assert!(to_cdn.iter().all(|seen| seen.authorization.is_none()), "the token crossed a redirect: {to_cdn:?}");
    assert_eq!(to_cdn[1].range.as_deref(), Some("bytes=512-"), "the range survives the hop");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn no_token_is_sent_when_none_was_chosen() {
    let (addr, state) = repo_server().await;
    let dir = temp_dir("no-token");
    downloader(addr).fetch(&entry(), &dir).await.expect("fetch");
    assert!(state.seen.lock().unwrap().iter().all(|seen| seen.authorization.is_none()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn the_token_never_appears_in_a_progress_line() {
    // 02 AC 7: the plain lines `model download` prints on stderr name the
    // URL — resuming, fetching, verifying — and never the header that
    // carried the token. The structured log is held to the same in
    // `tests/model_download_log.rs`, a binary of its own.
    let (addr, state) = repo_server().await;
    state.truncate_after.store(4096, Ordering::SeqCst);
    let dir = temp_dir("token-lines");
    let _ = downloader(addr).fetch(&entry(), &dir).await;
    state.truncate_after.store(0, Ordering::SeqCst);
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let collected = Arc::clone(&lines);
    let plain = Downloader::new(&format!("http://{addr}"), Some(ApiKey::new("tok-secret")))
        .expect("downloader")
        .reporting(Report::Lines(Arc::new(move |line: &str| collected.lock().unwrap().push(line.to_owned()))));
    plain.fetch(&entry(), &dir).await.expect("resumed");

    let lines = lines.lock().unwrap().clone();
    assert!(lines.iter().any(|line| line.contains(SIDECAR) && line.contains("kept")), "{lines:?}");
    assert!(lines.iter().any(|line| line.contains("resuming") && line.contains(&format!("http://{addr}/"))), "{lines:?}");
    assert!(lines.iter().any(|line| line.contains(ARTIFACT) && line.contains("verified")), "{lines:?}");
    for line in &lines {
        assert!(!line.contains("tok-secret"), "the token leaked: {line}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ── `ignis-server model download` ───────────────────────────────────────────

/// The filesystem `resolve_with` reads the catalog through: one file.
struct OneFile(PathBuf, String);

impl Files for OneFile {
    fn read(&self, path: &Path) -> std::io::Result<String> {
        if path == self.0 {
            Ok(self.1.clone())
        } else {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "not here"))
        }
    }

    fn write(&self, _: &Path, _: &str) -> std::io::Result<()> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "read-only"))
    }

    fn exists(&self, path: &Path) -> bool {
        path == self.0
    }
}

/// An operator catalog listing the served repo as `test-model`.
fn operator_catalog() -> OneFile {
    let file = |f: &CatalogFile| format!("      - {{ name: {}, bytes: {}, sha256: \"{}\" }}\n", f.name, f.bytes, f.sha256);
    let entry = entry();
    let text = format!(
        "models:\n  - id: test-model\n    repo: {REPO}\n    revision: \"{REVISION}\"\n    artifact: {ARTIFACT}\n    files:\n{}{}",
        file(&entry.files[0]),
        file(&entry.files[1])
    );
    OneFile(PathBuf::from("ops.catalog.yaml"), text)
}

/// `ignis-server <argv>` resolved as `main` resolves it, with no
/// environment but `HF_TOKEN` when given.
fn command(argv: &[&str], files: &OneFile) -> ModelCommand {
    let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    match resolve_with(&argv, |_| None, files).expect("resolves") {
        ConfigOutcome::Model(command) => command,
        other => panic!("expected a model command, got {other:?}"),
    }
}

/// Run `command` as `main` does: what it printed, what it reported, and its
/// exit code.
async fn ran(command: ModelCommand) -> (String, Vec<String>, i32) {
    let mut out = Vec::new();
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let collected = Arc::clone(&lines);
    let code = run(command, &mut out, Arc::new(move |line: &str| collected.lock().unwrap().push(line.to_owned()))).await;
    let lines = lines.lock().unwrap().clone();
    (String::from_utf8(out).unwrap(), lines, code)
}

#[tokio::test]
async fn model_download_fetches_an_entry_into_the_download_path_and_prints_its_artifact() {
    // 02 AC 12: through the same resolution a start takes (the catalog named
    // by a flag, the endpoint too); `download.enabled` off changes nothing,
    // nothing is asked, and this test binary has no `cuda`. The artifact's
    // path on stdout, progress as lines, exit 0.
    let (addr, _state) = repo_server().await;
    let dir = temp_dir("cli");
    let endpoint = format!("http://{addr}");
    let files = operator_catalog();
    let dir_text = dir.display().to_string();
    let argv = ["model", "download", "test-model", "--download-catalog", "ops.catalog.yaml", "--download-endpoint", &endpoint, "--download-path", &dir_text, "--download-enabled", "false"];

    let (stdout, progress, code) = ran(command(&argv, &files)).await;

    assert_eq!(code, 0, "{progress:?}");
    assert_eq!(stdout.trim(), dir.join(ARTIFACT).display().to_string());
    assert_eq!(std::fs::read(dir.join(ARTIFACT)).unwrap(), artifact_bytes());
    assert!(progress.iter().any(|line| line.starts_with("test-model:") && line.contains(&endpoint)), "{progress:?}");
    assert!(progress.iter().any(|line| line.contains(SIDECAR) && line.contains("verified")), "{progress:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn model_download_out_puts_it_elsewhere_and_no_id_fetches_the_configured_model() {
    // 02 AC 12: `--out` wins over `download.path`; with no id named, the
    // configured `model.id` is the one fetched.
    let (addr, _state) = repo_server().await;
    let out = temp_dir("cli-out");
    let endpoint = format!("http://{addr}");
    let files = operator_catalog();
    let out_text = out.display().to_string();
    let argv = ["model", "download", "--out", &out_text, "--model-id", "test-model", "--download-catalog", "ops.catalog.yaml", "--download-endpoint", &endpoint, "--download-path", "never-used"];

    let (stdout, progress, code) = ran(command(&argv, &files)).await;

    assert_eq!(code, 0, "{progress:?}");
    assert_eq!(stdout.trim(), out.join(ARTIFACT).display().to_string());
    assert!(out.join(SIDECAR).exists());
    assert!(!Path::new("never-used").exists());
    let _ = std::fs::remove_dir_all(&out);
}

#[tokio::test]
async fn model_download_exits_nonzero_when_an_entry_does_not_verify() {
    // 02 AC 12: a failure is a nonzero exit, its reason on the progress
    // stream; nothing unverified is printed as fetched.
    let (addr, state) = repo_server().await;
    state.extra_bytes.store(1, Ordering::SeqCst);
    let dir = temp_dir("cli-fail");
    let endpoint = format!("http://{addr}");
    let files = operator_catalog();
    let dir_text = dir.display().to_string();
    let argv = ["model", "download", "test-model", "--download-catalog", "ops.catalog.yaml", "--download-endpoint", &endpoint, "--download-path", &dir_text];

    let (stdout, progress, code) = ran(command(&argv, &files)).await;

    assert_ne!(code, 0);
    assert!(stdout.is_empty(), "{stdout}");
    assert!(progress.iter().any(|line| line.contains("test-model: failed") && line.contains("longer")), "{progress:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
