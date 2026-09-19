//! The model transfer (GitHub #234, spec
//! `.scratch/model-download/specs/01-model-download.md` AC3-AC5), driven
//! against a local `axum` server: no network, no GPU, no 19 GB.
//!
//! What is pinned here is what protects the operator from a file that is not
//! the model: the digest and the byte count are checked before anything is
//! renamed into place, the sidecar is fetched first, and an interrupted
//! transfer resumes from the `.part` instead of starting over.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Path as UrlPath, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;

use ignis_server::download::{Downloader, ModelEntry};

// ── the repo this test serves ───────────────────────────────────────────────

const ARTIFACT: &str = "test-model.ninfer";
const SIDECAR: &str = "test-model.ninfer.graft.json";
const REPO: &str = "ignis-test/model";
const SIDECAR_BODY: &[u8] = br#"{"recipe_id":"test"}"#;

/// The bytes the fake repo serves as the artifact: long enough that a resume
/// has something to resume from.
fn artifact_bytes() -> Vec<u8> {
    (0..64 * 1024u32).map(|i| (i % 251) as u8).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// A registry entry for the served artifact. `bytes`/`sha256` are the pins the
/// transfer checks what it wrote against, so a test tampers with them to make
/// the check fail.
fn entry(bytes: u64, sha256: &'static str) -> ModelEntry {
    ModelEntry {
        model: "test-model",
        repo: REPO,
        artifact_file: ARTIFACT,
        sidecar_file: SIDECAR,
        bytes,
        sha256,
    }
}

// ── the fake Hugging Face ───────────────────────────────────────────────────

#[derive(Default)]
struct Served {
    /// Every `Range` header the artifact route was sent (`None` = no header).
    ranges: Mutex<Vec<Option<String>>>,
    /// Artifact bytes actually written into responses.
    artifact_bytes_sent: AtomicUsize,
    /// Cut every artifact response off after this many bytes (0 = serve all).
    truncate_after: AtomicUsize,
    /// Ignore `Range` and always answer `200` with the whole body.
    ignore_range: AtomicUsize,
    /// Answer every ranged request `416`, as a source whose file is shorter
    /// than the range asked for does.
    range_not_satisfiable: AtomicUsize,
    /// Serve this many extra bytes past the body (0 = none).
    extra_bytes: AtomicUsize,
}

/// `GET /{owner}/{name}/resolve/main/{file}` — the route Hugging Face serves
/// a repo file under, with `Range` support.
async fn resolve(
    State(state): State<Arc<Served>>,
    UrlPath((owner, name, file)): UrlPath<(String, String, String)>,
    headers: HeaderMap,
) -> Response {
    if format!("{owner}/{name}") != REPO {
        return StatusCode::NOT_FOUND.into_response();
    }
    if file == SIDECAR {
        return (StatusCode::OK, SIDECAR_BODY.to_vec()).into_response();
    }
    if file != ARTIFACT {
        return StatusCode::NOT_FOUND.into_response();
    }
    let range = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_owned());
    state.ranges.lock().expect("ranges").push(range.clone());

    if range.is_some() && state.range_not_satisfiable.load(Ordering::SeqCst) == 1 {
        return StatusCode::RANGE_NOT_SATISFIABLE.into_response();
    }
    let mut body = artifact_bytes();
    body.extend(std::iter::repeat_n(
        0xABu8,
        state.extra_bytes.load(Ordering::SeqCst),
    ));
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
    if truncate_after > 0 {
        served.truncate(truncate_after);
    }
    state
        .artifact_bytes_sent
        .fetch_add(served.len(), Ordering::SeqCst);
    let status = if start > 0 && !ignore_range {
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    (status, Body::from(served)).into_response()
}

/// The fake repo on a loopback port, and the state a test steers it with.
async fn repo_server() -> (SocketAddr, Arc<Served>) {
    let state = Arc::new(Served::default());
    let app = Router::new()
        .route("/{owner}/{name}/resolve/main/{file}", get(resolve))
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

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ignis-download-{tag}-{}",
        std::process::id() as u64 * 1_000 + rand_suffix()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A per-call suffix, so two tests in the same process never share a
/// directory (no `rand` dependency for four digits).
fn rand_suffix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .subsec_nanos() as u64
        % 1_000
}

// ── the tests ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_clean_transfer_writes_the_artifact_and_its_sidecar() {
    // AC3/AC5: the sidecar lands next to the artifact under the name the
    // loader looks for, the artifact matches its pinned digest, and no
    // `.part` survives the transfer.
    let (addr, _state) = repo_server().await;
    let body = artifact_bytes();
    let digest: &'static str = Box::leak(sha256_hex(&body).into_boxed_str());
    let entry = entry(body.len() as u64, digest);
    let dir = temp_dir("clean");

    let downloader =
        Downloader::with_base(format!("http://{addr}")).expect("downloader");
    let path = downloader.fetch(&entry, &dir).await.expect("fetch");

    assert_eq!(path, dir.join(ARTIFACT));
    assert_eq!(std::fs::read(&path).expect("artifact"), body);
    assert_eq!(
        std::fs::read(dir.join(SIDECAR)).expect("sidecar"),
        SIDECAR_BODY
    );
    assert!(
        !dir.join(format!("{ARTIFACT}.part")).exists(),
        "the part file is renamed, not left behind"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_body_that_does_not_match_the_pinned_digest_is_refused() {
    // AC3: the bytes arrived intact, but they are not the model this build
    // knows — nothing is renamed into place, and the error says so.
    let (addr, _state) = repo_server().await;
    let body = artifact_bytes();
    let entry = entry(
        body.len() as u64,
        "0000000000000000000000000000000000000000000000000000000000000000",
    );
    let dir = temp_dir("tampered");

    let downloader =
        Downloader::with_base(format!("http://{addr}")).expect("downloader");
    let err = downloader
        .fetch(&entry, &dir)
        .await
        .expect_err("a wrong digest must refuse the transfer");

    let message = err.to_string();
    assert!(message.contains("sha256"), "{message}");
    assert!(!dir.join(ARTIFACT).exists(), "no artifact is left in place");
    assert!(
        !dir.join(format!("{ARTIFACT}.part")).exists(),
        "the rejected part file is removed, so the next start does not resume it"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_short_body_is_refused_before_the_digest_is_even_checked() {
    // AC3: a connection that dies mid-transfer leaves fewer bytes than the
    // pin names. The `.part` stays, so the next start resumes it.
    let (addr, state) = repo_server().await;
    let body = artifact_bytes();
    let digest: &'static str = Box::leak(sha256_hex(&body).into_boxed_str());
    let entry = entry(body.len() as u64, digest);
    let dir = temp_dir("short");
    state.truncate_after.store(4096, Ordering::SeqCst);

    let downloader =
        Downloader::with_base(format!("http://{addr}")).expect("downloader");
    let err = downloader
        .fetch(&entry, &dir)
        .await
        .expect_err("a short body must refuse the transfer");

    assert!(err.to_string().contains("4096"), "{err}");
    assert!(!dir.join(ARTIFACT).exists());
    assert_eq!(
        std::fs::metadata(dir.join(format!("{ARTIFACT}.part")))
            .expect("the part file survives an interrupted transfer")
            .len(),
        4096
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn an_interrupted_transfer_resumes_from_the_part_file() {
    // AC4: the second run asks for the rest with `Range`, hashes the bytes
    // already on disk, and ends with the same verified file — without
    // fetching the first 4 KiB twice.
    let (addr, state) = repo_server().await;
    let body = artifact_bytes();
    let digest: &'static str = Box::leak(sha256_hex(&body).into_boxed_str());
    let entry = entry(body.len() as u64, digest);
    let dir = temp_dir("resume");
    let downloader =
        Downloader::with_base(format!("http://{addr}")).expect("downloader");

    state.truncate_after.store(4096, Ordering::SeqCst);
    downloader
        .fetch(&entry, &dir)
        .await
        .expect_err("the first attempt is cut short");
    state.truncate_after.store(0, Ordering::SeqCst);
    let path = downloader.fetch(&entry, &dir).await.expect("resumed fetch");

    assert_eq!(std::fs::read(&path).expect("artifact"), body);
    let ranges = state.ranges.lock().expect("ranges").clone();
    assert_eq!(ranges.len(), 2, "one request per attempt");
    assert_eq!(ranges[0], None, "the first attempt asks for the whole file");
    assert_eq!(
        ranges[1].as_deref(),
        Some("bytes=4096-"),
        "the second asks only for what is missing"
    );
    assert_eq!(
        state.artifact_bytes_sent.load(Ordering::SeqCst),
        4096 + (body.len() - 4096),
        "the resumed bytes are transferred once each"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_server_that_ignores_the_range_restarts_the_transfer_cleanly() {
    // AC4, the other branch: answering `200` to a `Range` request means the
    // body starts at zero. Appending it to the part file would corrupt it, so
    // the transfer starts over — and still verifies.
    let (addr, state) = repo_server().await;
    let body = artifact_bytes();
    let digest: &'static str = Box::leak(sha256_hex(&body).into_boxed_str());
    let entry = entry(body.len() as u64, digest);
    let dir = temp_dir("norange");
    let downloader =
        Downloader::with_base(format!("http://{addr}")).expect("downloader");

    state.truncate_after.store(4096, Ordering::SeqCst);
    downloader.fetch(&entry, &dir).await.expect_err("cut short");
    state.truncate_after.store(0, Ordering::SeqCst);
    state.ignore_range.store(1, Ordering::SeqCst);
    let path = downloader.fetch(&entry, &dir).await.expect("restarted fetch");

    assert_eq!(std::fs::read(&path).expect("artifact"), body);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_source_longer_than_the_pin_is_discarded_mid_transfer() {
    // A body that is still arriving past the published size is a different
    // file, whatever its first bytes were: stop at the chunk that crosses the
    // line rather than write the rest, and leave nothing to resume.
    let (addr, state) = repo_server().await;
    let body = artifact_bytes();
    let digest: &'static str = Box::leak(sha256_hex(&body).into_boxed_str());
    let entry = entry(body.len() as u64, digest);
    let dir = temp_dir("toolong");
    state.extra_bytes.store(8192, Ordering::SeqCst);

    let downloader = Downloader::with_base(format!("http://{addr}")).expect("downloader");
    let err = downloader
        .fetch(&entry, &dir)
        .await
        .expect_err("a longer source must refuse the transfer");

    assert!(err.to_string().contains("longer"), "{err}");
    assert!(!dir.join(ARTIFACT).exists());
    assert!(
        !dir.join(format!("{ARTIFACT}.part")).exists(),
        "nothing is left for the next start to resume"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_partial_the_source_cannot_satisfy_is_discarded_instead_of_retried_forever() {
    // The source's file ends before the `.part` does (it was republished
    // shorter): the ranged request comes back `416`. Keeping the partial file
    // would send the same unsatisfiable range on every start from now on.
    let (addr, state) = repo_server().await;
    let body = artifact_bytes();
    let digest: &'static str = Box::leak(sha256_hex(&body).into_boxed_str());
    let entry = entry(body.len() as u64, digest);
    let dir = temp_dir("unsatisfiable");
    let downloader = Downloader::with_base(format!("http://{addr}")).expect("downloader");

    state.truncate_after.store(4096, Ordering::SeqCst);
    downloader.fetch(&entry, &dir).await.expect_err("cut short");
    state.truncate_after.store(0, Ordering::SeqCst);
    state.range_not_satisfiable.store(1, Ordering::SeqCst);
    let err = downloader
        .fetch(&entry, &dir)
        .await
        .expect_err("416 must refuse the transfer");
    assert!(err.to_string().contains("4096"), "{err}");
    assert!(
        !dir.join(format!("{ARTIFACT}.part")).exists(),
        "the unresumable part file is discarded"
    );

    // And the start after that one fetches the artifact whole.
    state.range_not_satisfiable.store(0, Ordering::SeqCst);
    let path = downloader.fetch(&entry, &dir).await.expect("clean fetch");
    assert_eq!(std::fs::read(&path).expect("artifact"), body);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_repo_without_the_sidecar_fails_before_the_artifact_is_fetched() {
    // AC5: the sidecar is a second-long request and the loader refuses a load
    // without it — so it is fetched first, and its failure costs nothing.
    let (addr, state) = repo_server().await;
    let body = artifact_bytes();
    let digest: &'static str = Box::leak(sha256_hex(&body).into_boxed_str());
    let mut entry = entry(body.len() as u64, digest);
    entry.sidecar_file = "test-model.ninfer.absent.json";
    let dir = temp_dir("nosidecar");

    let downloader =
        Downloader::with_base(format!("http://{addr}")).expect("downloader");
    let err = downloader
        .fetch(&entry, &dir)
        .await
        .expect_err("a missing sidecar must fail the transfer");

    assert!(err.to_string().contains("404"), "{err}");
    assert_eq!(
        state.artifact_bytes_sent.load(Ordering::SeqCst),
        0,
        "not one byte of the 19 GB body is fetched"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
