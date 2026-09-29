//! Shared by the Responses API's two test seams (GitHub #282): a scripted
//! template that turns chosen mock tokens into chosen text and records every
//! conversation it renders, a server over it, and — for the WebSocket seam —
//! a live listener and a `tokio-tungstenite` client.
//!
//! Included by `#[path]` from `responses_http.rs` and `responses_socket.rs`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use ignis_core::{Compute, ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{
    ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection,
};
use ignis_server::telemetry::FixedClock;
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;
use serde_json::Value as JsonValue;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

pub const MODEL: &str = "test-model";

/// The token a template with boundaries renders after the generation opener.
pub const GENERATION_PROMPT: TokenId = 0x7FFF_FFF0;

/// What every response's `created_at` reads, so events are reproducible.
pub const CREATED_AT_MS: u64 = 1_790_000_000_000;

/// One conversation as the template received it.
#[derive(Clone, Debug, PartialEq)]
pub struct Rendered {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<JsonValue>,
}

/// The placeholder template's tokens, a scripted decode, and a record of
/// every render. With `boundaries`, it reports the structure a real template
/// does: the system block (its leading `system` messages' words), the last
/// user turn, and the generation opener, followed — as a thinking template's
/// is — by one token of generation prompt.
#[derive(Default)]
pub struct Script {
    decode: HashMap<TokenId, &'static str>,
    boundaries: bool,
    pub seen: Mutex<Vec<Rendered>>,
}

impl Script {
    /// Tokens decode to `decode`'s text, and to `?` otherwise.
    pub fn new(decode: HashMap<TokenId, &'static str>) -> Arc<Self> {
        Arc::new(Self { decode, ..Self::default() })
    }

    /// [`Script::new`], reporting the prompt's structure.
    pub fn with_boundaries(decode: HashMap<TokenId, &'static str>) -> Arc<Self> {
        Arc::new(Self { decode, boundaries: true, ..Self::default() })
    }

    pub fn seen(&self) -> Vec<Rendered> {
        self.seen.lock().unwrap().clone()
    }
}

/// The server's handle on a shared [`Script`].
struct Handle(Arc<Script>);

impl TemplateProvider for Handle {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        self.0.seen.lock().unwrap().push(Rendered { messages: messages.to_vec(), tools: tools.to_vec() });
        let mut rendered = SimpleTemplateProvider.apply_chat_template(messages, options, tools)?;
        if self.0.boundaries {
            let words = |m: &ChatMessage| m.content.text().split_whitespace().count() as u32;
            let system: u32 = messages.iter().take_while(|m| m.role == "system").map(words).sum();
            rendered.system_block_tokens = Some(system).filter(|&n| n > 0);
            let last_user = messages.iter().rposition(|m| m.role == "user");
            rendered.user_turn_tokens = last_user.map(|at| messages[..at].iter().map(words).sum());
            rendered.opener_tokens = Some(rendered.tokens.len() as u32);
            rendered.tokens.push(GENERATION_PROMPT);
        }
        Ok(rendered)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens.iter().map(|t| self.0.decode.get(t).copied().unwrap_or("?")).collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(Decoder(self.0.decode.clone()))
    }
    // `decoder_starts_in_reasoning` stays the trait default: a request with
    // thinking on starts in the reasoning channel, as a real template's does.
}

struct Decoder(HashMap<TokenId, &'static str>);

impl TokenDecoder for Decoder {
    fn push(&mut self, token: TokenId) -> String {
        self.0.get(&token).copied().unwrap_or("?").to_owned()
    }
    fn finish(&mut self) -> String {
        String::new()
    }
}

/// A server over `compute` and `script`, with a fixed wall clock.
pub fn server(script: &Arc<Script>, config: SchedulerConfig, compute: Arc<dyn Compute>) -> Server {
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..config }, compute);
    Server::new(Engine::new(Box::new(scheduler)), Box::new(Handle(Arc::clone(script))))
        .with_request_timeout(Duration::from_secs(5))
        .with_wall_clock(Arc::new(FixedClock::new(CREATED_AT_MS)))
}

/// The events of a `text/event-stream` body: each `event:` line's name, and
/// the `data:` line after it parsed.
pub fn sse_events(body: &str) -> Vec<(String, JsonValue)> {
    let mut events = Vec::new();
    let mut name = None;
    for line in body.lines() {
        if let Some(event) = line.strip_prefix("event:") {
            name = Some(event.trim().to_owned());
        } else if let Some(data) = line.strip_prefix("data:") {
            let data = serde_json::from_str(data.trim()).expect("each data line is one JSON event");
            events.push((name.take().expect("an event line precedes its data"), data));
        }
    }
    events
}

// ── the WebSocket seam ───────────────────────────────────────────────────

/// A server serving on a random localhost port until dropped.
pub struct Live {
    pub addr: std::net::SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for Live {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}

/// Serve `server` on a fresh listener.
pub async fn live(server: Server) -> Live {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(server.serve_on_until(listener, None, async {
        let _ = stopped.await;
    }));
    Live { addr, stop: Some(stop) }
}

pub type Socket = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Open `/v1/responses` with `headers` on the upgrade; the socket and the
/// handshake response, or the refusal.
pub async fn open(
    live: &Live,
    headers: &[(&str, &str)],
) -> Result<(Socket, tokio_tungstenite::tungstenite::handshake::client::Response), tokio_tungstenite::tungstenite::Error> {
    let mut request = format!("ws://{}/v1/responses", live.addr).into_client_request().unwrap();
    for (name, value) in headers {
        request.headers_mut().append(
            tokio_tungstenite::tungstenite::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            value.parse().unwrap(),
        );
    }
    tokio_tungstenite::connect_async(request).await
}

/// Open `/v1/responses` on an open server.
pub async fn socket(live: &Live) -> Socket {
    open(live, &[]).await.expect("the upgrade succeeds").0
}

/// Send one JSON event.
pub async fn send(socket: &mut Socket, event: JsonValue) {
    socket.send(Message::Text(event.to_string().into())).await.expect("send");
}

/// The next text frame, raw, within a generous bound (a hang fails loudly).
pub async fn next_frame(socket: &mut Socket) -> String {
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(10), socket.next())
            .await
            .expect("a frame within 10 s")
            .expect("the socket is open")
            .expect("a frame");
        match frame {
            Message::Text(text) => return text.to_string(),
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("unexpected frame {other:?}"),
        }
    }
}

/// The next event, parsed.
pub async fn next_event(socket: &mut Socket) -> JsonValue {
    serde_json::from_str(&next_frame(socket).await).expect("a JSON event")
}

/// Events up to and including the one `until` accepts.
pub async fn events_until(socket: &mut Socket, until: impl Fn(&JsonValue) -> bool) -> Vec<JsonValue> {
    let mut events = Vec::new();
    loop {
        let event = next_event(socket).await;
        let done = until(&event);
        events.push(event);
        if done {
            return events;
        }
    }
}

/// Whether `event` ends a response or is an `error`.
pub fn terminal(event: &JsonValue) -> bool {
    matches!(
        event["type"].as_str(),
        Some("response.completed" | "response.incomplete" | "response.failed" | "error")
    )
}
