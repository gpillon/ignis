//! Media acquisition (GitHub #179, spec `docs/specs/vision/01-image-input.md`
//! §Wire contract): the `MediaAcquirer` seam the chat handler calls before
//! admission, driven on the CPU against a local image server and the real
//! vision processor (ADR 0006: no GPU, no network beyond loopback).
//!
//! - URL policy: a loopback fetch needs `--media-allow-private-network`; a
//!   redirect to a refused address is refused; an oversize body, a body that
//!   never finishes and an HTTP error each map to their code.
//! - Budgets: a single image at each limit is accepted and one over is
//!   refused; several images whose total passes the vision or byte budget
//!   lose their oldest, in whole blocks (GitHub #673), never prepared.
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
use support::media::{data_uri, png, processor, shade, until, Gated};

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

/// `probe` with the limit `set` to `at` accepts the 64x64 image, and with
/// `at - 1` refuses it.
async fn boundary(set: fn(&mut ProcessorOptions, u64), at: u64) {
    let image = data_uri(&png(64, 64));
    let mut exact = limits();
    set(&mut exact, at);
    let accepted = acquire(&acquirer(exact, MediaPolicy::new(false, 0)), &[&image]).await;
    assert_eq!(accepted.map(|a| a.media.len()).map_err(|r| r.message), Ok(1));
    let mut under = limits();
    set(&mut under, at - 1);
    assert_refused(acquire(&acquirer(under, MediaPolicy::new(false, 0)), &[&image]).await, "media_budget_exceeded");
}

/// The older of two images: 64 x 64, 4 merged tokens.
fn older() -> Vec<u8> {
    shade(64, 64, 40)
}

/// The newer: 96 x 64, 6 merged tokens.
fn newer() -> Vec<u8> {
    shade(96, 64, 200)
}

/// GitHub #673: with the limit `set` to the pair's own total `at`, both
/// images are seen; one under it, the older is dropped and the newer kept
/// whole -- admitted, not refused.
async fn pair_boundary(set: fn(&mut ProcessorOptions, u64), at: u64) {
    let (older, newer) = (data_uri(&older()), data_uri(&newer()));
    let mut exact = limits();
    set(&mut exact, at);
    let both = acquire(&acquirer(exact, MediaPolicy::new(false, 0)), &[&older, &newer]).await.unwrap();
    assert_eq!((both.media.len(), both.dropped, both.stats.dropped_items), (2, 0, 0));
    let mut under = limits();
    set(&mut under, at - 1);
    let one = acquire(&acquirer(under, MediaPolicy::new(false, 0)), &[&older, &newer]).await.unwrap();
    assert_eq!((one.media.len(), one.dropped), (1, 1), "the older is dropped");
    assert_eq!(one.media[0].source_pixels, (96, 64), "the newer is kept");
    assert_eq!(one.media[0].grid, Grid { t: 1, h: 4, w: 6 }, "whole");
    assert_eq!((one.stats.items, one.stats.vision_tokens, one.stats.media_bytes), (1, 6, self::newer().len() as u64));
    assert_eq!(
        (one.stats.dropped_items, one.stats.dropped_vision_tokens, one.stats.dropped_bytes),
        (1, 4, self::older().len() as u64)
    );
    assert_eq!(one.stats.source_pixels, Some((96, 64)), "the first image the model sees");
}

#[tokio::test]
async fn the_byte_budget_accepts_one_image_at_its_limit_and_refuses_it_one_over() {
    boundary(|o, n| o.max_encoded_media_bytes = n, png(64, 64).len() as u64).await;
}

#[tokio::test]
async fn two_images_one_byte_past_the_byte_budget_lose_the_older() {
    pair_boundary(|o, n| o.max_encoded_media_bytes = n, (older().len() + newer().len()) as u64).await;
}

#[tokio::test]
async fn the_decoded_pixel_budget_accepts_exactly_its_limit() {
    boundary(|o, n| o.max_decoded_pixels = n, 64 * 64).await;
}

#[tokio::test]
async fn the_raw_patch_budget_accepts_one_image_at_its_limit_and_refuses_it_one_over() {
    boundary(|o, n| o.max_raw_patches = n, 16).await;
}

#[tokio::test]
async fn two_images_one_raw_patch_past_the_budget_lose_the_older() {
    pair_boundary(|o, n| o.max_raw_patches = n, 4 * (4 + 6)).await;
}

#[tokio::test]
async fn two_images_one_token_past_the_vision_budget_lose_the_older() {
    pair_boundary(|o, n| o.max_vision_tokens = n, 4 + 6).await;
}

/// A request's 64 x 64 images, `n` of them, each its own bytes.
fn history(n: usize) -> Vec<String> {
    (0..n).map(|i| data_uri(&shade(64, 64, i as u8))).collect()
}

/// GitHub #673: a history that grows by one image a request, past a
/// 16-token budget. A block takes images up to half of it, two of these 4-token
/// images, so the oldest go two at a time and the dropped set only grows: the
/// counts below are worked by hand from the rule, block by block.
#[tokio::test]
async fn a_growing_history_drops_its_oldest_images_a_whole_block_at_a_time() {
    let acquirer = acquirer(ProcessorOptions { max_vision_tokens: 16, ..limits() }, MediaPolicy::new(false, 1 << 20));
    let images = history(12);
    let mut dropped = Vec::new();
    for n in 1..=images.len() {
        let urls: Vec<&str> = images[..n].iter().map(String::as_str).collect();
        let acquired = acquire(&acquirer, &urls).await.unwrap_or_else(|r| panic!("{n} images: {}", r.message));
        assert!(acquired.stats.vision_tokens <= 16, "{n} images: {:?}", acquired.stats);
        assert_eq!(acquired.media.len() + acquired.dropped, n);
        dropped.push(acquired.dropped);
    }
    assert_eq!(dropped, [0, 0, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8]);
}

/// An image past half the budget is a block of its own, and the newest
/// image is never dropped, however large.
#[tokio::test]
async fn an_image_past_half_the_budget_is_its_own_block_and_the_newest_is_kept() {
    let acquirer = acquirer(ProcessorOptions { max_vision_tokens: 16, ..limits() }, MediaPolicy::new(false, 0));
    let (small, large, whole) = (|l| data_uri(&shade(64, 64, l)), |l| data_uri(&shade(128, 96, l)), data_uri(&shade(128, 128, 9)));
    // 12 tokens then 4: within the 16 together, nothing dropped.
    let fits = acquire(&acquirer, &[&large(1), &small(2)]).await.unwrap();
    assert_eq!(fits.dropped, 0);
    // 4, 4, then 12: the 12 starts its own block, so the two 4s go together.
    let one = acquire(&acquirer, &[&small(3), &small(4), &large(5)]).await.unwrap();
    assert_eq!((one.dropped, one.stats.vision_tokens), (2, 12));
    // 4, 12, 4, 4: the 12 is a block between two others, dropped whole
    // after the first, which leaves the last block, the two last 4s.
    let alone = acquire(&acquirer, &[&small(6), &large(7), &small(8), &small(10)]).await.unwrap();
    assert_eq!((alone.dropped, alone.stats.vision_tokens), (2, 8));
    // 4, 4, 4, then the whole budget at once: everything before it goes.
    let newest = acquire(&acquirer, &[&small(11), &small(12), &small(13), &whole]).await.unwrap();
    assert_eq!((newest.dropped, newest.stats.vision_tokens), (3, 16));
    assert_eq!(newest.media[0].grid, Grid { t: 1, h: 8, w: 8 });
}

/// A dropped image is never prepared: the preparer sees no call for it, and
/// the cache holds nothing for it, so sent again alone it is a miss.
#[tokio::test]
async fn a_dropped_image_is_never_prepared_nor_cached() {
    let gated = Gated::new(true, processor(ProcessorOptions { max_vision_tokens: 9, ..limits() }));
    let acquirer = MediaAcquirer::new(gated.clone(), ProcessorOptions { max_vision_tokens: 9, ..limits() }, MediaPolicy::new(false, 1 << 20));
    let (older, newer) = (data_uri(&older()), data_uri(&newer()));
    let acquired = acquire(&acquirer, &[&older, &newer]).await.unwrap();
    assert_eq!(acquired.dropped, 1);
    assert_eq!(gated.builds.load(Ordering::SeqCst), 1, "only the kept image was prepared");
    assert_eq!((acquired.stats.cache_hits, acquired.stats.cache_misses), (0, 1));
    let alone = acquire(&acquirer, &[&older]).await.unwrap();
    assert_eq!((alone.dropped, alone.stats.cache_hits, alone.stats.cache_misses), (0, 0, 1), "never cached");
    assert_eq!(gated.builds.load(Ordering::SeqCst), 2);
    let again = acquire(&acquirer, &[&newer]).await.unwrap();
    assert_eq!((again.stats.cache_hits, again.stats.cache_misses), (1, 0), "the kept one was");
}

/// What is still refused is refused for any image, dropped or not: an image
/// no size can fit, an undecodable one, and one alone past the byte budget.
#[tokio::test]
async fn an_image_that_would_be_refused_alone_is_refused_wherever_it_is() {
    let acquirer = acquirer(ProcessorOptions { max_vision_tokens: 9, ..limits() }, MediaPolicy::new(false, 0));
    let (strip, garbage) = (data_uri(&png(201, 1)), data_uri(b"not an image"));
    let newer = data_uri(&newer());
    for bad in [&strip, &garbage] {
        let refused = assert_refused(acquire(&acquirer, &[bad, &newer, &newer]).await, "invalid_media");
        assert!(refused.message.starts_with("message 0 content part 1: "), "{}", refused.message);
    }
    let big = data_uri(&png(64, 64));
    let mut small = limits();
    small.max_encoded_media_bytes = png(64, 64).len() as u64 - 1;
    let refused = assert_refused(acquire(&acquirer_with(small), &[&big, &data_uri(&shade(32, 32, 1))]).await, "media_budget_exceeded");
    assert!(refused.message.starts_with("message 0 content part 1: "), "{}", refused.message);
}

fn acquirer_with(limits: ProcessorOptions) -> MediaAcquirer {
    acquirer(limits, MediaPolicy::new(false, 0))
}

/// GitHub #248: one image over the budget is shrunk to it, not refused, and
/// the stats keep what it would have been.
#[tokio::test]
async fn a_single_image_over_the_vision_token_budget_is_shrunk_to_it() {
    let image = data_uri(&png(64, 64));
    let mut under = limits();
    under.max_vision_tokens = 3;
    let acquired = acquire(&acquirer(under, MediaPolicy::new(false, 0)), &[&image]).await.unwrap();
    assert_eq!(acquired.media.len(), 1);
    assert_eq!((acquired.stats.vision_tokens, acquired.stats.natural_vision_tokens), (1, 4));
    assert_eq!(acquired.media[0].source_pixels, (64, 64));
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
