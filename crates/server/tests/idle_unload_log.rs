//! The log lines of idle unload (spec model-switch/02 AC 2 and 6): the
//! unload is `ignis.model.unloaded`, once, and the load a request begins
//! after it `ignis.model.load_started` then `ignis.model.reloaded` — neither
//! of them a switch — while a switch out of `unloaded` is logged as any
//! switch is; an unload a client asks for (GitHub #723) is the same line,
//! without the idle time. A binary of its own with one test, which installs the
//! process's subscriber: the unload's line is written on the blocking thread
//! that tears the model down, which a subscriber scoped to the test's thread
//! never sees.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::model_switch::Switcher;
use ignis_server::{ModelStatus, Server};
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

/// Every line `log` holds named `name`.
fn named(log: &MemorySink, name: &str) -> Vec<Value> {
    log.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("json"))
        .filter(|event| event["event_name"] == name)
        .collect()
}

/// An event's field, top level or under `attributes`.
fn field(event: &Value, name: &str) -> Value {
    event.get(name).or_else(|| event["attributes"].get(name)).cloned().unwrap_or(Value::Null)
}

/// Let `server` unload its model after a second, and wait until `log` holds
/// the `count`th `ignis.model.unloaded`; then the knob is zero again, so
/// nothing unloads under what the test does next.
async fn unloaded(server: &Server, log: &MemorySink, count: usize) {
    server.update_live(|live| live.idle_unload = Duration::from_secs(1));
    let deadline = Instant::now() + Duration::from_secs(10);
    while named(log, "ignis.model.unloaded").len() < count {
        assert!(Instant::now() < deadline, "never unloaded a {count}th time");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    assert_eq!(*server.status(), ModelStatus::Unloaded);
}

/// A request to `uri` with `body`, its status once answered. The response is
/// dropped here: a response held, its body read or not, is a request in use.
async fn post(app: &axum::Router, uri: &str, body: &Value) -> StatusCode {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(request).await.unwrap().status()
}

#[tokio::test]
async fn the_unload_and_the_load_after_it_are_logged_and_neither_is_a_switch() {
    let log = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone())))
        .expect("the binary's only subscriber");
    let loader = MockLoader::new();
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, Duration::from_secs(30)));
    let app = server.app();
    let _clock = server.spawn_idle_clock(Duration::from_millis(50)).expect("a clock");

    unloaded(&server, &log, 1).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let unloads = named(&log, "ignis.model.unloaded");
    assert_eq!(unloads.len(), 1, "logged once: {unloads:?}");
    assert_eq!(field(&unloads[0], "model"), json!("mock-a"));
    assert_eq!(field(&unloads[0], "idle_secs"), json!(1));

    // The load a request begins: started and ended as a load, not a switch.
    let chat = json!({ "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 });
    assert_eq!(post(&app, "/v1/chat/completions", &chat).await, StatusCode::OK);
    let started = named(&log, "ignis.model.load_started");
    assert_eq!(started.len(), 1, "{started:?}");
    assert_eq!((field(&started[0], "from"), field(&started[0], "to")), (json!("mock-a"), json!("mock-a")), "{}", started[0]);
    let reloaded = named(&log, "ignis.model.reloaded");
    assert_eq!(reloaded.len(), 1, "{reloaded:?}");
    assert_eq!(field(&reloaded[0], "model"), json!("mock-a"));
    for timing in ["took_ms", "load_ms", "warm_up_ms"] {
        assert!(field(&reloaded[0], timing).is_u64(), "{timing}: {}", reloaded[0]);
    }
    for switch in ["ignis.model.switch_started", "ignis.model.switched"] {
        assert!(named(&log, switch).is_empty(), "neither is a switch: {switch}");
    }

    // A switch out of `unloaded` is a switch: started and ended as one.
    unloaded(&server, &log, 2).await;
    let switch = json!({ "artifact": "mock-b.ninfer", "model": "mock-b" });
    assert_eq!(post(&app, "/v1/models/switch", &switch).await, StatusCode::ACCEPTED);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(*server.status(), ModelStatus::Serving) {
        assert!(Instant::now() < deadline, "the switch never landed: {:?}", server.status());
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let switch_started = named(&log, "ignis.model.switch_started");
    assert_eq!(switch_started.len(), 1, "{switch_started:?}");
    assert_eq!((field(&switch_started[0], "from"), field(&switch_started[0], "to")), (json!("mock-a"), json!("mock-b")));
    assert_eq!(named(&log, "ignis.model.switched").len(), 1);
    assert_eq!(named(&log, "ignis.model.load_started").len(), 1, "no load_started for a switch");
    assert_eq!(named(&log, "ignis.model.reloaded").len(), 1, "no reloaded for a switch");

    // An unload a client asks for (GitHub #723): the same line, logged before
    // the answer, with no idle time; the load after it is a load as before.
    assert_eq!(post(&app, "/v1/ignis/models/unload", &json!({ "model": "mock-b" })).await, StatusCode::OK);
    let unloads = named(&log, "ignis.model.unloaded");
    assert_eq!(unloads.len(), 3, "{unloads:?}");
    assert_eq!(field(&unloads[2], "model"), json!("mock-b"));
    assert_eq!(field(&unloads[2], "idle_secs"), Value::Null, "asked for, not idle: {}", unloads[2]);
    assert!(field(&unloads[2], "teardown_ms").is_u64(), "{}", unloads[2]);
    assert_eq!(post(&app, "/v1/chat/completions", &chat).await, StatusCode::OK);
    assert_eq!(named(&log, "ignis.model.reloaded").len(), 2, "the load after it is logged as a load");
    assert_eq!(named(&log, "ignis.model.switch_started").len(), 1, "neither is a switch");

    // Unloaded again, and the next load refused before it reaches the GPU:
    // `reload_failed` at the prepare stage, still not a switch.
    unloaded(&server, &log, 4).await;
    loader.refuse("mock-b");
    assert_eq!(post(&app, "/v1/chat/completions", &chat).await, StatusCode::SERVICE_UNAVAILABLE);
    let failed = named(&log, "ignis.model.reload_failed");
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert_eq!((field(&failed[0], "model"), field(&failed[0], "stage")), (json!("mock-b"), json!("prepare")), "{}", failed[0]);
    assert!(named(&log, "ignis.model.switch_failed").is_empty(), "not a switch either");
}
