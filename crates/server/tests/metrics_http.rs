//! GitHub #89 / ADR 0017: Prometheus metrics through the public routers,
//! backed by a real `ConcreteScheduler` over the deterministic `MockCompute`
//! (CPU-only, ADR 0006). With `--metrics`, `GET /metrics` is served by its
//! own listener (`Server::metrics_app`), never by the API's; the Playground
//! reads the same exposition at `/ui/metrics` on the API listener, under the
//! API key when one is set. Values move with the request lifecycle the
//! engine's telemetry consumer observes.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use ignis_core::mock::GatedCompute;
use ignis_core::{Compute, ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::config::ApiKey;
use ignis_server::engine::Engine;
use ignis_server::playground::Assets;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

const MODEL: &str = "test-model";
const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";
const KEY: &str = "sk-metrics";
/// The Playground's asset table; its content does not matter here.
const PAGE: Assets = &[("index.html", b"<!doctype html>")];

fn server_over(compute: Arc<dyn Compute>) -> Server {
    server_with(SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() }, compute)
}

fn server_with(config: SchedulerConfig, compute: Arc<dyn Compute>) -> Server {
    let scheduler = ConcreteScheduler::with_config(config, compute);
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
}

fn server() -> Server {
    server_over(Arc::new(MockCompute::new()))
}

/// A metrics-enabled server's two routers: the API listener's and the
/// metrics listener's.
fn apps(server: Server) -> (axum::Router, axum::Router) {
    let metrics = server.metrics_app().expect("metrics are on");
    (server.app(), metrics)
}

async fn send(app: &axum::Router, uri: &str, auth: Option<&str>) -> axum::response::Response {
    let mut request = Request::builder().uri(uri);
    if let Some(auth) = auth {
        request = request.header(header::AUTHORIZATION, auth);
    }
    app.clone().oneshot(request.body(Body::empty()).unwrap()).await.unwrap()
}

async fn get(app: &axum::Router, uri: &str) -> axum::response::Response {
    send(app, uri, None).await
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn scrape(metrics: &axum::Router) -> String {
    let response = get(metrics, "/metrics").await;
    assert_eq!(response.status(), StatusCode::OK);
    body_text(response).await
}

/// A non-streaming chat completion naming `model`.
async fn chat(app: &axum::Router, model: &str, max_tokens: u32) -> axum::response::Response {
    let body = serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": max_tokens,
        "stream": false
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(request).await.unwrap()
}

/// A successful completion; returns its `usage.completion_tokens`.
async fn complete(app: &axum::Router, max_tokens: u32) -> u64 {
    let response = chat(app, MODEL, max_tokens).await;
    assert_eq!(response.status(), StatusCode::OK);
    let json: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
    json["usage"]["completion_tokens"].as_u64().expect("usage.completion_tokens")
}

/// Every sample line of an exposition: `(name, labels, value)`.
fn samples(text: &str) -> Vec<(String, Vec<(String, String)>, String)> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| {
            let (series, value) = line.rsplit_once(' ').expect("`series value`");
            let (name, labels) = match series.split_once('{') {
                Some((name, labels)) => (name, labels.trim_end_matches('}')),
                None => (series, ""),
            };
            let labels = labels
                .split(',')
                .filter(|pair| !pair.is_empty())
                .map(|pair| {
                    let (key, value) = pair.split_once('=').expect("`key=\"value\"`");
                    (key.to_owned(), value.trim_matches('"').to_owned())
                })
                .collect();
            (name.to_owned(), labels, value.to_owned())
        })
        .collect()
}

/// The value of the sample `name`, with `state` as its only label when given.
fn value(text: &str, name: &str, state: Option<&str>) -> u64 {
    samples(text)
        .into_iter()
        .find(|(n, labels, _)| {
            n == name
                && match state {
                    Some(state) => labels.iter().any(|(k, v)| k == "state" && v == state),
                    None => labels.is_empty(),
                }
        })
        .unwrap_or_else(|| panic!("no {name} {state:?} in:\n{text}"))
        .2
        .parse()
        .unwrap()
}

/// Scrapes until `done` holds for the exposition, within a few seconds: the
/// telemetry consumer runs asynchronously, so a value lands after the HTTP
/// response that caused it, never before.
async fn scrape_until(metrics: &axum::Router, done: impl Fn(&str) -> bool) -> String {
    let mut last = String::new();
    let settled = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            last = scrape(metrics).await;
            if done(&last) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(settled.is_ok(), "the projection never settled; last scrape:\n{last}");
    last
}

// ── which listener serves what ──────────────────────────────────────────────

#[tokio::test]
async fn without_metrics_there_is_no_metrics_listener_and_no_route() {
    let server = server().with_playground(PAGE);
    assert!(server.metrics_app().is_none());
    let app = server.app();
    for uri in ["/metrics", "/ui/metrics"] {
        assert_eq!(get(&app, uri).await.status(), StatusCode::NOT_FOUND, "{uri}");
    }
    assert_eq!(get(&app, "/v1/models").await.status(), StatusCode::OK);
}

#[tokio::test]
async fn the_api_listener_never_serves_metrics_at_the_root() {
    let (api, _) = apps(server().with_playground(PAGE).with_metrics());
    assert_eq!(get(&api, "/metrics").await.status(), StatusCode::NOT_FOUND);
    assert_eq!(get(&api, "/v1/models").await.status(), StatusCode::OK);
}

#[tokio::test]
async fn the_metrics_listener_serves_only_the_exposition_in_the_text_format() {
    let (_, metrics) = apps(server().with_metrics());
    let response = get(&metrics, "/metrics").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get(header::CONTENT_TYPE).unwrap(), CONTENT_TYPE);
    let text = body_text(response).await;
    for (name, kind) in [
        ("ignis_build_info", "gauge"),
        ("ignis_scheduler_requests", "gauge"),
        ("ignis_requests_accepted_total", "counter"),
        ("ignis_requests_completed_total", "counter"),
        ("ignis_requests_cancelled_total", "counter"),
        ("ignis_generated_tokens_total", "counter"),
        ("ignis_decoded_tokens_total", "counter"),
        ("ignis_kv_cache_evictions_total", "counter"),
        ("ignis_prefix_reused_tokens_total", "counter"),
        ("ignis_requests_rejected_total", "counter"),
        ("ignis_request_ttft_seconds", "histogram"),
        ("ignis_request_duration_seconds", "histogram"),
    ] {
        assert!(text.contains(&format!("\n# TYPE {name} {kind}\n")), "{name}:\n{text}");
        assert!(text.contains(&format!("# HELP {name} ")), "{name}:\n{text}");
    }
    assert!(
        text.contains(&format!("ignis_build_info{{version=\"{}\"}} 1\n", env!("CARGO_PKG_VERSION"))),
        "{text}"
    );
    for uri in ["/v1/models", "/ui/metrics", "/ui/"] {
        assert_eq!(get(&metrics, uri).await.status(), StatusCode::NOT_FOUND, "{uri}");
    }
}

/// The metrics listener is not the API: it asks for no key, even when the
/// API does. It is kept off the network by its bind address, and `--expose`
/// never tunnels it (ADR 0017, ADR 0028).
#[tokio::test]
async fn the_metrics_listener_needs_no_api_key() {
    let (api, metrics) = apps(server().with_api_key(ApiKey::new(KEY)).with_metrics());
    assert_eq!(get(&metrics, "/metrics").await.status(), StatusCode::OK);
    assert_eq!(get(&api, "/v1/models").await.status(), StatusCode::UNAUTHORIZED);
}

// ── the Playground's copy: /ui/metrics ──────────────────────────────────────

#[tokio::test]
async fn the_playground_reads_the_same_exposition_at_ui_metrics() {
    let (api, metrics) = apps(server().with_playground(PAGE).with_metrics());
    complete(&api, 2).await;
    let expected = scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total", None) == 1).await;

    let response = get(&api, "/ui/metrics").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get(header::CONTENT_TYPE).unwrap(), CONTENT_TYPE);
    assert_eq!(body_text(response).await, expected);
}

#[tokio::test]
async fn with_an_api_key_ui_metrics_requires_it() {
    let (api, _) = apps(server().with_playground(PAGE).with_api_key(ApiKey::new(KEY)).with_metrics());
    assert_eq!(get(&api, "/ui/metrics").await.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        send(&api, "/ui/metrics", Some("Bearer sk-wrong")).await.status(),
        StatusCode::UNAUTHORIZED
    );
    let response = send(&api, "/ui/metrics", Some(&format!("Bearer {KEY}"))).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(body_text(response).await.contains("# TYPE ignis_build_info gauge"));
    // The page itself stays reachable without a key (ADR 0026).
    assert_eq!(get(&api, "/ui/").await.status(), StatusCode::OK);
}

#[tokio::test]
async fn ui_metrics_exists_only_with_both_the_playground_and_metrics() {
    let (api, _) = apps(server().with_metrics());
    assert_eq!(get(&api, "/ui/metrics").await.status(), StatusCode::NOT_FOUND);
    // With the Playground but no metrics, `/ui/metrics` is just an unknown
    // Playground path.
    let api = server().with_playground(PAGE).app();
    assert_eq!(get(&api, "/ui/metrics").await.status(), StatusCode::NOT_FOUND);
}

// ── values ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn completed_requests_move_the_counters_and_leave_no_request_in_flight() {
    let (api, metrics) = apps(server().with_metrics());
    let before = scrape(&metrics).await;
    assert_eq!(value(&before, "ignis_requests_accepted_total", None), 0);
    assert_eq!(value(&before, "ignis_requests_completed_total", None), 0);
    assert_eq!(value(&before, "ignis_generated_tokens_total", None), 0);

    let first = complete(&api, 4).await;
    let text = scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total", None) == 1).await;
    assert_eq!(value(&text, "ignis_requests_accepted_total", None), 1);
    assert_eq!(value(&text, "ignis_generated_tokens_total", None), first);

    let second = complete(&api, 3).await;
    let text = scrape_until(&metrics, |t| {
        value(t, "ignis_requests_completed_total", None) == 2
            && value(t, "ignis_scheduler_requests", Some("running")) == 0
    })
    .await;
    assert_eq!(value(&text, "ignis_requests_accepted_total", None), 2);
    assert_eq!(value(&text, "ignis_generated_tokens_total", None), first + second);
    // Every token of a completed request was also counted as it was decoded.
    assert_eq!(value(&text, "ignis_decoded_tokens_total", None), first + second);
    assert_eq!(value(&text, "ignis_requests_cancelled_total", None), 0);
    assert_eq!(value(&text, "ignis_scheduler_requests", Some("waiting")), 0);
}

#[tokio::test]
async fn a_rejected_submission_is_not_counted_as_accepted() {
    let (api, metrics) = apps(server().with_metrics());
    let response = chat(&api, "no-such-model", 2).await;
    assert!(response.status().is_client_error(), "{}", response.status());

    // A later accepted request settles the projection, so a stray count
    // from the rejection would be visible by then.
    complete(&api, 2).await;
    let text = scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total", None) == 1).await;
    assert_eq!(value(&text, "ignis_requests_accepted_total", None), 1, "{text}");
}

/// A client that disconnects mid-generation cancels its request, and the
/// scheduler releases a cancelled request without a `Done`. The request must
/// still leave the gauges — `ignis_scheduler_requests` is the current state,
/// not every request ever seen — and count as cancelled.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_disconnect_takes_the_request_out_of_the_gauges() {
    let (gated, controller) = GatedCompute::new(Arc::new(MockCompute::new()));
    let (api, metrics) = apps(server_over(gated.clone() as Arc<dyn Compute>).with_metrics());

    // Hold the first decode step, so the stream is still generating when
    // the client goes away.
    gated.arm();
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "keep decoding" }],
        "max_tokens": 4096,
        "stream": true
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = api.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let controller = tokio::task::spawn_blocking(move || {
        controller.wait_entered();
        controller
    })
    .await
    .unwrap();
    // Disconnect while the step is held, so the cancel is queued before the
    // model thread can finish anything; then let it go.
    drop(response);
    tokio::task::spawn_blocking(move || controller.release()).await.unwrap();

    // A later request settles the projection: once it has completed, the
    // abandoned one must be gone from both gauges and counted as cancelled.
    complete(&api, 2).await;
    let text = scrape_until(&metrics, |t| {
        value(t, "ignis_requests_completed_total", None) == 1
            && value(t, "ignis_requests_cancelled_total", None) == 1
            && value(t, "ignis_scheduler_requests", Some("waiting")) == 0
            && value(t, "ignis_scheduler_requests", Some("running")) == 0
    })
    .await;
    // Every accepted request is accounted for exactly once.
    assert_eq!(value(&text, "ignis_requests_accepted_total", None), 2, "{text}");
}

/// The value of the sample `name` carrying exactly the one label `key=value`.
fn labelled(text: &str, name: &str, key: &str, label: &str) -> u64 {
    samples(text)
        .into_iter()
        .find(|(n, labels, _)| n == name && labels.len() == 1 && labels[0] == (key.to_owned(), label.to_owned()))
        .unwrap_or_else(|| panic!("no {name}{{{key}={label}}} in:\n{text}"))
        .2
        .parse()
        .unwrap()
}

fn rejected(text: &str, reason: &str) -> u64 {
    labelled(text, "ignis_requests_rejected_total", "reason", reason)
}

/// GitHub #90: every rejection is counted on the HTTP side, after the submit
/// call returned its error — so it is already in the very next scrape, with
/// no telemetry fact to wait for — and never as accepted.
#[tokio::test]
async fn rejections_are_counted_by_fixed_reason() {
    let (api, metrics) = apps(server().with_metrics());
    assert_eq!(chat(&api, "no-such-model", 2).await.status(), StatusCode::NOT_FOUND);
    // Longer than the per-sequence context (GitHub #166): a request that
    // can never fit, counted with the oversized ones.
    assert_eq!(chat(&api, MODEL, 100_000).await.status(), StatusCode::BAD_REQUEST);

    let text = scrape(&metrics).await;
    assert_eq!(rejected(&text, "unknown_model"), 1, "{text}");
    assert_eq!(rejected(&text, "oversized"), 1, "{text}");
    assert_eq!(rejected(&text, "full"), 0, "{text}");
    assert_eq!(value(&text, "ignis_requests_accepted_total", None), 0, "{text}");

    // Larger than the whole KV pool: 413.
    let tiny_pool = SchedulerConfig { model: MODEL.into(), kv_capacity_pages: 4, ..SchedulerConfig::default() };
    let (api, metrics) = apps(server_with(tiny_pool, Arc::new(MockCompute::new())).with_metrics());
    assert_eq!(chat(&api, MODEL, 1000).await.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let text = scrape(&metrics).await;
    assert_eq!(rejected(&text, "oversized"), 1, "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_engine_counts_a_full_rejection() {
    let (gated, controller) = GatedCompute::new(Arc::new(MockCompute::new()));
    let one_lane = SchedulerConfig { model: MODEL.into(), max_in_flight: 1, ..SchedulerConfig::default() };
    let (api, metrics) = apps(server_with(one_lane, gated.clone() as Arc<dyn Compute>).with_metrics());

    // Hold the only request mid-decode, one step at a time, so the engine
    // stays full.
    gated.arm();
    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hold the lane" }],
        "max_tokens": 4096,
        "stream": true
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let held = api.clone().oneshot(request).await.unwrap();
    assert_eq!(held.status(), StatusCode::OK);
    let mut controller = tokio::task::spawn_blocking(move || {
        controller.wait_entered();
        controller
    })
    .await
    .unwrap();

    // The model thread takes the second submit between two held steps,
    // while the first request is still in flight.
    let second = tokio::spawn({
        let api = api.clone();
        async move { chat(&api, MODEL, 2).await.status() }
    });
    while !second.is_finished() {
        gated.arm();
        controller = tokio::task::spawn_blocking(move || {
            controller.release();
            controller.wait_entered();
            controller
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(second.await.unwrap(), StatusCode::SERVICE_UNAVAILABLE);
    let text = scrape(&metrics).await;
    assert_eq!(rejected(&text, "full"), 1, "{text}");

    drop(held);
    tokio::task::spawn_blocking(move || controller.release()).await.unwrap();
}

/// GitHub #90: a completed request is one TTFT and one request-duration
/// observation.
#[tokio::test]
async fn a_completed_request_is_observed_by_both_latency_histograms() {
    let (api, metrics) = apps(server().with_metrics());
    complete(&api, 3).await;
    let text = scrape_until(&metrics, |t| value(t, "ignis_request_duration_seconds_count", None) == 1).await;
    assert_eq!(value(&text, "ignis_request_ttft_seconds_count", None), 1, "{text}");
    for name in ["ignis_request_ttft_seconds_bucket", "ignis_request_duration_seconds_bucket"] {
        assert_eq!(labelled(&text, name, "le", "+Inf"), 1, "{text}");
    }
}

/// GitHub #90: scrapes running concurrently with each other and with the
/// workload all succeed, and the workload is fully accounted for once they
/// settle.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_scrapes_neither_fail_nor_disturb_the_workload() {
    let (api, metrics) = apps(server().with_metrics());
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let scrapers: Vec<_> = (0..16)
        .map(|_| {
            let (metrics, stop) = (metrics.clone(), Arc::clone(&stop));
            tokio::spawn(async move {
                let mut scrapes = 0u32;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) || scrapes == 0 {
                    let text = scrape(&metrics).await;
                    assert!(text.ends_with('\n') && text.contains("# TYPE ignis_build_info gauge"));
                    scrapes += 1;
                    tokio::task::yield_now().await;
                }
                scrapes
            })
        })
        .collect();
    let requests: Vec<_> = (0..8u32).map(|i| {
        let api = api.clone();
        tokio::spawn(async move { complete(&api, 2 + i % 3).await })
    }).collect();
    let mut generated = 0;
    for request in requests {
        generated += request.await.unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for scraper in scrapers {
        assert!(scraper.await.unwrap() > 0);
    }

    let text = scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total", None) == 8).await;
    assert_eq!(value(&text, "ignis_requests_accepted_total", None), 8, "{text}");
    assert_eq!(value(&text, "ignis_generated_tokens_total", None), generated, "{text}");
}

/// GitHub #90 / ADR 0017, over real sockets: scrapers that never read their
/// responses, or hang up mid-request, stall only their own connections. The
/// API keeps completing requests and the telemetry consumer keeps the
/// projection current for a well-behaved scraper.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_and_disconnected_scrapers_cannot_hold_back_inference_or_telemetry() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let server = server().with_metrics();
    let api_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let metrics_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (api_addr, metrics_addr) = (api_listener.local_addr().unwrap(), metrics_listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(server.serve_on_until(api_listener, Some(metrics_listener), async {
        let _ = stopped.await;
    }));

    // Slow: hundreds of pipelined scrapes per connection, never read, so
    // each connection's socket buffers fill and its writer stalls.
    let mut slow = Vec::new();
    for _ in 0..8 {
        let mut stream = tokio::net::TcpStream::connect(metrics_addr).await.unwrap();
        let burst = "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n".repeat(500);
        // The server stops reading once its writes back up; a partial write
        // is enough to wedge the connection.
        let _ = tokio::time::timeout(Duration::from_millis(200), stream.write_all(burst.as_bytes())).await;
        slow.push(stream);
    }
    // Disconnected: half a request, then gone.
    for _ in 0..8 {
        let mut stream = tokio::net::TcpStream::connect(metrics_addr).await.unwrap();
        stream.write_all(b"GET /metr").await.unwrap();
        drop(stream);
    }

    let body = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 3,
        "stream": false
    })
    .to_string();
    let outcome = tokio::time::timeout(Duration::from_secs(10), async {
        for _ in 0..3 {
            let mut stream = tokio::net::TcpStream::connect(api_addr).await.unwrap();
            let request = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(request.as_bytes()).await.unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).await.unwrap();
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        }
        loop {
            let mut stream = tokio::net::TcpStream::connect(metrics_addr).await.unwrap();
            stream.write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").await.unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).await.unwrap();
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            if response.contains("\nignis_requests_completed_total 3\n") {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(outcome.is_ok(), "stalled scrapers held back the API or the telemetry consumer");

    drop(slow);
    let _ = stop.send(());
    let _ = tokio::time::timeout(Duration::from_secs(5), serving).await;
}

#[tokio::test]
async fn labels_stay_within_the_bounded_sets() {
    let (api, metrics) = apps(server().with_metrics());
    complete(&api, 2).await;
    let text = scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total", None) == 1).await;

    let names: BTreeSet<String> = samples(&text).into_iter().map(|(name, _, _)| name).collect();
    let expected: BTreeSet<String> = [
        "ignis_build_info",
        "ignis_scheduler_requests",
        "ignis_requests_accepted_total",
        "ignis_requests_completed_total",
        "ignis_requests_cancelled_total",
        "ignis_generated_tokens_total",
        "ignis_decoded_tokens_total",
        "ignis_kv_cache_evictions_total",
        "ignis_prefix_reused_tokens_total",
        "ignis_retained_reused_tokens_total",
        "ignis_retained_state_hits_total",
        "ignis_retained_state_misses_total",
        "ignis_retained_state_spills_total",
        "ignis_retained_state_discards_total",
        "ignis_retained_state_restores_total",
        "ignis_requests_rejected_total",
        "ignis_request_ttft_seconds_bucket",
        "ignis_request_ttft_seconds_sum",
        "ignis_request_ttft_seconds_count",
        "ignis_request_duration_seconds_bucket",
        "ignis_request_duration_seconds_sum",
        "ignis_request_duration_seconds_count",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(names, expected, "{text}");

    const TTFT_LE: &[&str] =
        &["0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60", "120", "300", "+Inf"];
    const DURATION_LE: &[&str] =
        &["0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60", "120", "300", "600", "+Inf"];
    for (name, labels, _) in samples(&text) {
        let allowed: &[(&str, &[&str])] = match name.as_str() {
            "ignis_build_info" => &[("version", &[env!("CARGO_PKG_VERSION")])],
            "ignis_scheduler_requests" => &[("state", &["waiting", "running"])],
            "ignis_requests_rejected_total" => &[("reason", &["full", "unknown_model", "oversized"])],
            "ignis_request_ttft_seconds_bucket" => &[("le", TTFT_LE)],
            "ignis_request_duration_seconds_bucket" => &[("le", DURATION_LE)],
            // GitHub #190: one series per residency tier, never per entry.
            name if name.starts_with("ignis_retained_") => &[("tier", &["device", "kv_ram"])],
            _ => &[],
        };
        assert_eq!(labels.len(), allowed.len(), "{name} {labels:?}");
        for (key, value) in &labels {
            let (_, values) = allowed
                .iter()
                .find(|(k, _)| k == key)
                .unwrap_or_else(|| panic!("{name} carries an unbounded label `{key}`"));
            assert!(values.contains(&value.as_str()), "{name}{{{key}={value}}}");
        }
    }
}
