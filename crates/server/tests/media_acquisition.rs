//! Media acquisition (GitHub #179, spec `docs/specs/vision/01-image-input.md`
//! §Wire contract): the `MediaAcquirer` seam the chat handler calls before
//! admission, driven on the CPU against a local image server and the real
//! vision processor (ADR 0006: no GPU, no network beyond loopback).
//!
//! - URL policy: a loopback fetch needs `--media-allow-private-network`; a
//!   redirect to a refused address is refused; an oversize body, a body that
//!   never finishes and an HTTP error each map to their code.
//! - Budgets: each at its limit is accepted and one over is refused.
//! - Cache: the same image twice is a hit the second time; two concurrent
//!   identical misses build once.
//! - Cancellation: dropping the acquisition stops the preparation.

use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr};
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

use ignis_artifact::vision::{Grid, ProcessorOptions};
use ignis_server::media::{AcquiredMedia, MediaAcquirer, MediaPolicy, MediaRejection};
use ignis_server::template::{ChatMessage, ContentPart, MessageContent};

#[path = "support/mod.rs"]
mod support;
use support::media::{data_uri, png, processor, until, Gated};

// ── fixtures ────────────────────────────────────────────────────────────────

/// Limits far above the 64x64 test image, so a test lowers only the one it
/// probes.
fn limits() -> ProcessorOptions {
    ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 1 << 20,
        max_decoded_pixels: 1 << 20,
        max_raw_patches: 1 << 16,
        max_vision_tokens: 1 << 14,
    }
}

fn acquirer(limits: ProcessorOptions, policy: MediaPolicy) -> MediaAcquirer {
    MediaAcquirer::new(Arc::new(processor(limits.clone())), limits, policy)
}

/// One user message: a text part, then one image part per URL.
fn messages(urls: &[&str]) -> Vec<ChatMessage> {
    let mut parts = vec![ContentPart { kind: Some("text".to_owned()), text: Some("look".to_owned()), url: None, cache_control: None }];
    parts.extend(urls.iter().map(|url| ContentPart {
        kind: Some("image_url".to_owned()),
        text: None,
        url: Some((*url).to_owned()),
        cache_control: None,
    }));
    let mut message = ChatMessage::text("user", "");
    message.content = MessageContent::Parts(parts);
    vec![message]
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

async fn acquire(acquirer: &MediaAcquirer, urls: &[&str]) -> Result<AcquiredMedia, MediaRejection> {
    acquirer.acquire(&messages(urls), deadline()).await
}

fn assert_refused(result: Result<AcquiredMedia, MediaRejection>, code: &str) -> MediaRejection {
    let rejection = result.expect_err("expected a refusal");
    assert_eq!((rejection.status, rejection.code), (400, code), "{}", rejection.message);
    rejection
}

// ── the image server ────────────────────────────────────────────────────────

/// A body that never yields a byte.
struct Stalled;

impl futures_core::Stream for Stalled {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

/// A chunked body of `n` bytes, with no `Content-Length`.
struct Chunks(usize);

impl futures_core::Stream for Chunks {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.0 == 0 {
            return Poll::Ready(None);
        }
        let n = self.0.min(256);
        self.0 -= n;
        Poll::Ready(Some(Ok(Bytes::from(vec![0u8; n]))))
    }
}

/// Serve test images on `127.0.0.1`, returning the port.
async fn image_server() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let redirect = |to: String| move || async move { (StatusCode::FOUND, [(header::LOCATION, to)]) };
    let app = Router::new()
        .route("/image.png", get(|| async { png(64, 64) }))
        .route("/redirect-local", get(redirect("/image.png".to_owned())))
        .route("/redirect-private", get(redirect(format!("http://127.0.0.2:{port}/image.png"))))
        .route("/big", get(|| async { vec![0u8; 2048] }))
        .route("/big-chunked", get(|| async { Body::from_stream(Chunks(2048)) }))
        .route("/stalled", get(|| async { Body::from_stream(Stalled).into_response() }))
        .route("/missing", get(|| async { StatusCode::NOT_FOUND }));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    port
}

// ── URL policy ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_loopback_fetch_succeeds_only_with_the_private_network_opt_in() {
    let port = image_server().await;
    let url = format!("http://127.0.0.1:{port}/image.png");

    let refused = assert_refused(acquire(&acquirer(limits(), MediaPolicy::new(false, 0)), &[&url]).await, "invalid_media");
    assert!(refused.message.contains("message 0 content part 1"), "{}", refused.message);
    assert!(refused.message.contains("disallowed network addresses"), "{}", refused.message);

    let acquired = acquire(&acquirer(limits(), MediaPolicy::new(true, 0)), &[&url]).await.unwrap();
    assert_eq!(acquired.media.len(), 1);
    assert_eq!(acquired.media[0].grid, Grid { t: 1, h: 4, w: 4 });
    assert_eq!(acquired.stats.items, 1);
    assert_eq!(acquired.stats.vision_tokens, 4);
    assert_eq!(acquired.stats.media_bytes, png(64, 64).len() as u64);
}

#[tokio::test]
async fn private_addresses_are_refused_by_name_too() {
    // `localhost` resolves to loopback only: nothing left to connect to.
    let port = image_server().await;
    let url = format!("http://localhost:{port}/image.png");
    assert_refused(acquire(&acquirer(limits(), MediaPolicy::new(false, 0)), &[&url]).await, "invalid_media");
}

#[tokio::test]
async fn a_redirect_is_followed_and_rechecked_so_one_to_a_refused_address_is_refused() {
    let port = image_server().await;
    // Loopback is allowed, except the address the redirect points at.
    let refuse_127_0_0_2 = Arc::new(|ip: IpAddr| ip != IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)));
    let policy = MediaPolicy::new(false, 0).with_address_filter(refuse_127_0_0_2);
    let acquirer = acquirer(limits(), policy);

    let followed = acquire(&acquirer, &[&format!("http://127.0.0.1:{port}/redirect-local")]).await.unwrap();
    assert_eq!(followed.media[0].grid, Grid { t: 1, h: 4, w: 4 });

    let refused = assert_refused(
        acquire(&acquirer, &[&format!("http://127.0.0.1:{port}/redirect-private")]).await,
        "invalid_media",
    );
    assert!(refused.message.contains("disallowed network addresses"), "{}", refused.message);
}

#[tokio::test]
async fn an_oversize_body_is_over_the_byte_budget_with_or_without_a_length() {
    let port = image_server().await;
    let mut small = limits();
    small.max_encoded_media_bytes = 1024;
    let acquirer = acquirer(small, MediaPolicy::new(true, 0));
    for path in ["big", "big-chunked"] {
        let url = format!("http://127.0.0.1:{port}/{path}");
        assert_refused(acquire(&acquirer, &[&url]).await, "media_budget_exceeded");
    }
}

#[tokio::test]
async fn a_body_that_never_finishes_is_a_fetch_timeout() {
    let port = image_server().await;
    let mut policy = MediaPolicy::new(true, 0);
    policy.fetch_timeout = Duration::from_millis(200);
    let url = format!("http://127.0.0.1:{port}/stalled");
    assert_refused(acquire(&acquirer(limits(), policy), &[&url]).await, "media_fetch_timeout");
}

#[tokio::test]
async fn an_http_error_is_a_fetch_failure() {
    let port = image_server().await;
    let url = format!("http://127.0.0.1:{port}/missing");
    let refused = assert_refused(acquire(&acquirer(limits(), MediaPolicy::new(true, 0)), &[&url]).await, "media_fetch_failed");
    assert!(refused.message.contains("HTTP 404"), "{}", refused.message);
}

#[tokio::test]
async fn credentials_and_other_schemes_are_invalid_media() {
    let acquirer = acquirer(limits(), MediaPolicy::new(true, 0));
    for url in ["http://user:secret@127.0.0.1:1/image.png", "file:///etc/passwd", "ftp://127.0.0.1/a.png"] {
        assert_refused(acquire(&acquirer, &[url]).await, "invalid_media");
    }
}

// ── budgets ─────────────────────────────────────────────────────────────────

/// `probe` with the limit `set` to `at` accepts the 64x64 image (twice in
/// one request when `pair`), and with `at - 1` refuses it.
async fn boundary(set: fn(&mut ProcessorOptions, u64), at: u64, pair: bool) {
    let image = data_uri(&png(64, 64));
    let urls: Vec<&str> = if pair { vec![&image, &image] } else { vec![&image] };
    let mut exact = limits();
    set(&mut exact, at);
    let accepted = acquire(&acquirer(exact, MediaPolicy::new(false, 0)), &urls).await;
    assert_eq!(accepted.map(|a| a.media.len()).map_err(|r| r.message), Ok(urls.len()));
    let mut under = limits();
    set(&mut under, at - 1);
    assert_refused(acquire(&acquirer(under, MediaPolicy::new(false, 0)), &urls).await, "media_budget_exceeded");
}

#[tokio::test]
async fn the_byte_budget_accepts_exactly_its_limit_across_the_request() {
    let bytes = png(64, 64).len() as u64;
    boundary(|o, n| o.max_encoded_media_bytes = n, bytes, false).await;
    boundary(|o, n| o.max_encoded_media_bytes = n, 2 * bytes, true).await;
}

#[tokio::test]
async fn the_decoded_pixel_budget_accepts_exactly_its_limit() {
    boundary(|o, n| o.max_decoded_pixels = n, 64 * 64, false).await;
}

#[tokio::test]
async fn the_raw_patch_budget_accepts_exactly_its_limit_across_the_request() {
    boundary(|o, n| o.max_raw_patches = n, 16, false).await;
    boundary(|o, n| o.max_raw_patches = n, 32, true).await;
}

#[tokio::test]
async fn the_vision_token_budget_accepts_exactly_its_limit_across_the_request() {
    boundary(|o, n| o.max_vision_tokens = n, 4, false).await;
    boundary(|o, n| o.max_vision_tokens = n, 8, true).await;
}

// ── cache, single-flight, cancellation ─────────────────────────────────────

#[tokio::test]
async fn the_same_image_sent_twice_is_a_cache_hit_the_second_time() {
    let gated = Gated::new(true, processor(limits()));
    let acquirer = MediaAcquirer::new(gated.clone(), limits(), MediaPolicy::new(false, 1 << 20));
    let image = data_uri(&png(64, 64));

    let first = acquire(&acquirer, &[&image]).await.unwrap();
    assert_eq!((first.stats.cache_hits, first.stats.cache_misses), (0, 1));
    let second = acquire(&acquirer, &[&image]).await.unwrap();
    assert_eq!((second.stats.cache_hits, second.stats.cache_misses), (1, 0));
    assert_eq!(second.stats.preprocess_seconds, 0.0);
    assert_eq!(second.media, first.media);
    assert_eq!(gated.builds.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_zero_cache_retains_nothing() {
    let gated = Gated::new(true, processor(limits()));
    let acquirer = MediaAcquirer::new(gated.clone(), limits(), MediaPolicy::new(false, 0));
    let image = data_uri(&png(64, 64));
    for _ in 0..2 {
        let acquired = acquire(&acquirer, &[&image]).await.unwrap();
        assert_eq!((acquired.stats.cache_hits, acquired.stats.cache_misses), (0, 1));
    }
    assert_eq!(gated.builds.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_concurrent_identical_misses_build_once() {
    let gated = Gated::new(false, processor(limits()));
    let acquirer = Arc::new(MediaAcquirer::new(gated.clone(), limits(), MediaPolicy::new(false, 0)));
    let image = data_uri(&png(64, 64));
    let request = |acquirer: Arc<MediaAcquirer>, image: String| {
        tokio::spawn(async move { acquirer.acquire(&messages(&[&image]), deadline()).await })
    };

    let first = request(acquirer.clone(), image.clone());
    until("the first build to start", || gated.builds.load(Ordering::SeqCst) == 1).await;
    let second = request(acquirer.clone(), image.clone());
    until("the second request to join", || acquirer.waiting_requests() == 2).await;
    gated.open.store(true, Ordering::SeqCst);

    let (first, second) = (first.await.unwrap().unwrap(), second.await.unwrap().unwrap());
    assert_eq!(gated.builds.load(Ordering::SeqCst), 1);
    assert_eq!((first.stats.cache_misses, first.stats.cache_hits), (1, 0));
    assert_eq!((second.stats.cache_misses, second.stats.cache_hits), (0, 1));
    assert_eq!(first.media, second.media);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_the_acquisition_stops_the_preparation() {
    let gated = Gated::new(false, processor(limits()));
    let acquirer = Arc::new(MediaAcquirer::new(gated.clone(), limits(), MediaPolicy::new(false, 1 << 20)));
    let image = data_uri(&png(64, 64));
    let task = {
        let acquirer = acquirer.clone();
        tokio::spawn(async move { acquirer.acquire(&messages(&[&image]), deadline()).await })
    };
    until("the build to start", || gated.builds.load(Ordering::SeqCst) == 1).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    until("the build to observe the cancellation", || gated.observed_cancel.load(Ordering::SeqCst)).await;
    until("the flight to end", || acquirer.waiting_requests() == 0).await;
    // Nothing half-built was retained: the next request builds afresh.
    gated.open.store(true, Ordering::SeqCst);
    let acquired = acquire(&acquirer, &[&data_uri(&png(64, 64))]).await.unwrap();
    assert_eq!(acquired.stats.cache_misses, 1);
    assert_eq!(gated.builds.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_passed_deadline_is_a_request_timeout() {
    let gated = Gated::new(false, processor(limits()));
    let acquirer = MediaAcquirer::new(gated, limits(), MediaPolicy::new(false, 0));
    let image = data_uri(&png(64, 64));
    let rejection = acquirer
        .acquire(&messages(&[&image]), Instant::now() + Duration::from_millis(50))
        .await
        .expect_err("the gate never opens");
    assert_eq!((rejection.status, rejection.code), (504, "request_timeout"));
}
