//! GitHub #81 / ADR 0012: the HTTP-ingress root span (`tower-http`'s
//! `TraceLayer`, `api.rs::router`'s `RootSpanMaker`) records the real
//! `RequestId` once the scheduler assigns one — proving the `Empty` +
//! later `Span::record` wiring actually fires on the real request path
//! (`chat_completions`), not just in an isolated unit test of the maker.
//!
//! This runs on the default (current-thread) `#[tokio::test]` runtime
//! deliberately, holding the subscriber active for the whole `.await` via
//! `tracing::dispatcher::set_default`'s guard: the HTTP task's own span
//! recording happens on the calling thread even though the scheduler's
//! actual admission/prefill/decode work runs on the engine's separate
//! model thread (GitHub #69) — `Engine::submit`'s `.await` only waits on a
//! channel, it does not move the rest of the handler to another thread. A
//! thread-local override therefore reaches every span this handler opens
//! or records into, without needing a process-wide global subscriber.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing::field::{Field, Visit};
use tracing::span::{Id, Record};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

const MODEL: &str = "test-model";

fn app() -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider)).app()
}

#[derive(Default)]
struct RequestIdVisitor(Option<u64>);

impl Visit for RequestIdVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "request_id" {
            self.0 = Some(value);
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

/// Captures every `request_id` recorded (via `Span::record`, `on_record`)
/// on any span seen while active — the root HTTP span declares it `Empty`
/// at creation, so only `on_record` (not `on_new_span`) ever sees it.
struct RootSpanProbe(Arc<Mutex<Vec<u64>>>);

impl<S> Layer<S> for RootSpanProbe
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_record(&self, _id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut visitor = RequestIdVisitor::default();
        values.record(&mut visitor);
        if let Some(id) = visitor.0 {
            self.0.lock().unwrap().push(id);
        }
    }
}

#[tokio::test]
async fn the_http_root_span_records_the_assigned_request_id() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let probe = RootSpanProbe(recorded.clone());
    let dispatch = tracing::Dispatch::new(tracing_subscriber::registry().with(probe));
    let _guard = tracing::dispatcher::set_default(&dispatch);

    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 1,
    });
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let response = app().oneshot(request).await.unwrap();
    assert_eq!(response.status(), 200);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let id_field = value["id"].as_str().expect("chat completion id");
    let expected: u64 = id_field
        .strip_prefix("chatcmpl-")
        .expect("id has the chatcmpl- prefix")
        .parse()
        .expect("the suffix is the numeric RequestId");

    drop(_guard);
    let recorded = recorded.lock().unwrap();
    assert_eq!(
        recorded.as_slice(),
        [expected],
        "the root span must record exactly the request's own assigned id"
    );
}
