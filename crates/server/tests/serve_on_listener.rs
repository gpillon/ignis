//! `Server::serve_on`: serving on a listener the caller already bound — what
//! `--expose` needs, since the tunnel is opened on the bound port before the
//! server starts serving (ADR 0028). The tunnel itself needs the network and
//! is checked by hand; this covers the seam `main` uses.

use std::sync::Arc;

use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::config::ApiKey;
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const KEY: &str = "sk-test-key";

fn server() -> Server {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_api_key(ApiKey::new(KEY))
}

/// One `GET /v1/models` over a fresh connection; the status line and body.
async fn get(addr: std::net::SocketAddr, auth: Option<&str>) -> String {
    get_path(addr, "/v1/models", auth).await
}

/// One HTTP/1.1 `GET path` over a fresh connection; the status line and body.
async fn get_path(addr: std::net::SocketAddr, path: &str, auth: Option<&str>) -> String {
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let auth = auth.map(|a| format!("Authorization: {a}\r\n")).unwrap_or_default();
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{auth}Connection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.expect("write");
    let mut response = String::new();
    stream.read_to_string(&mut response).await.expect("read");
    response
}

#[tokio::test]
async fn a_server_serves_on_a_listener_bound_before_it_and_stops_on_request() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    // The port is known before serving starts: this is where `main` opens
    // the tunnel.
    assert_eq!(ignis_server::expose::origin_port(addr), Ok(addr.port()));

    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(server().serve_on_until(listener, async {
        let _ = stopped.await;
    }));

    let refused = get(addr, None).await;
    assert!(refused.starts_with("HTTP/1.1 401"), "{refused}");
    let served = get(addr, Some(&format!("Bearer {KEY}"))).await;
    assert!(served.starts_with("HTTP/1.1 200"), "{served}");
    assert!(served.contains("test-model"), "{served}");

    stop.send(()).expect("server still running");
    serving.await.expect("join").expect("graceful stop");
}

/// GitHub #89 / ADR 0017: with `--metrics`, the exposition is served on its
/// own listener, without the API key; the API listener does not serve it at
/// the root; one shutdown stops both.
#[tokio::test]
async fn metrics_are_served_on_their_own_listener_and_stop_with_the_api() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let metrics_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind metrics");
    let (addr, metrics_addr) = (listener.local_addr().unwrap(), metrics_listener.local_addr().unwrap());

    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(server().with_metrics().serve_on_with_metrics_until(
        listener,
        Some(metrics_listener),
        async {
            let _ = stopped.await;
        },
    ));

    let scraped = get_path(metrics_addr, "/metrics", None).await;
    assert!(scraped.starts_with("HTTP/1.1 200"), "{scraped}");
    assert!(scraped.contains("# TYPE ignis_build_info gauge"), "{scraped}");
    let on_api = get_path(addr, "/metrics", Some(&format!("Bearer {KEY}"))).await;
    assert!(on_api.starts_with("HTTP/1.1 404"), "{on_api}");
    let served = get(addr, Some(&format!("Bearer {KEY}"))).await;
    assert!(served.starts_with("HTTP/1.1 200"), "{served}");

    stop.send(()).expect("server still running");
    serving.await.expect("join").expect("graceful stop");
    assert!(
        tokio::net::TcpStream::connect(metrics_addr).await.is_err(),
        "the metrics listener closed with the API"
    );
}
