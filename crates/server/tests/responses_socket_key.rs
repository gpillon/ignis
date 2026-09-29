//! The Responses WebSocket's key check (GitHub #282, spec responses-api/01,
//! acceptance 6): the upgrade takes the server's key as a bearer header or as
//! an `openai-insecure-api-key.<key>` subprotocol entry, never selects that
//! entry back, and never writes the key to the log.
//!
//! A binary of its own: it captures the log with a thread-scoped subscriber,
//! which only sees everything when no other test shares the process.

#[path = "support/responses.rs"]
mod responses;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ignis_core::{MockCompute, SchedulerConfig};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::config::ApiKey;
use serde_json::{json, Value as JsonValue};
use tokio_tungstenite::tungstenite::Error as WsError;
use tracing_subscriber::layer::SubscriberExt;

use responses::{live, next_event, open, send, server, terminal, Script, Socket, MODEL};

const KEY: &str = "sk-socket-test-key";

fn short() -> JsonValue {
    json!({ "type": "response.create", "model": MODEL, "input": "hi", "max_output_tokens": 1, "enable_thinking": false })
}

/// The type of the response's terminal event.
async fn ends(socket: &mut Socket) -> JsonValue {
    loop {
        let event = next_event(socket).await;
        if terminal(&event) {
            return event["type"].clone();
        }
    }
}

fn status_of(refused: WsError) -> u16 {
    match refused {
        WsError::Http(response) => response.status().as_u16(),
        other => panic!("expected an HTTP refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn the_upgrade_is_keyed_by_bearer_or_by_subprotocol_and_never_echoes_the_key() {
    let sink = Arc::new(MemorySink::new());
    let _logs = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(sink.clone())));
    let server = server(&Script::new(HashMap::new()), SchedulerConfig::default(), Arc::new(MockCompute::new()))
        .with_api_key(ApiKey::new(KEY));
    let live = live(server).await;
    let credential = format!("openai-insecure-api-key.{KEY}");

    assert_eq!(status_of(open(&live, &[]).await.unwrap_err()), 401);
    let wrong = "responses, openai-insecure-api-key.nope".to_owned();
    assert_eq!(status_of(open(&live, &[("Sec-WebSocket-Protocol", &wrong)]).await.unwrap_err()), 401);

    let bearer = format!("Bearer {KEY}");
    let (mut by_header, handshake) = open(&live, &[("Authorization", &bearer)]).await.expect("bearer opens");
    assert!(handshake.headers().get("sec-websocket-protocol").is_none(), "nothing offered, nothing selected");
    send(&mut by_header, short()).await;
    assert_eq!(ends(&mut by_header).await, "response.incomplete");

    let offered = format!("responses, {credential}");
    let (mut by_protocol, handshake) =
        open(&live, &[("Sec-WebSocket-Protocol", &offered)]).await.expect("the subprotocol key opens");
    assert_eq!(handshake.headers()["sec-websocket-protocol"], "responses", "the credential is never selected");
    send(&mut by_protocol, short()).await;
    assert_eq!(ends(&mut by_protocol).await, "response.incomplete");
    drop((by_header, by_protocol));

    // The request log's lines are written off the requests' own path: wait
    // for both requests' to be there to search.
    let done = || sink.lines().iter().filter(|l| l.contains("ignis.request.done")).count();
    let logged = tokio::time::timeout(Duration::from_secs(10), async {
        while done() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(logged.is_ok(), "both requests were logged: {:?}", sink.lines());
    let logged = sink.lines().join("\n");
    assert!(!logged.contains(KEY), "the key reached the log:\n{logged}");
}
