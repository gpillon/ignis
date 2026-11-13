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
use ignis_core::{Compute, ConcreteScheduler, MockCompute, RequestId, SchedulerConfig, TokenId};
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

/// Where [`Script::encoding`]'s character tokens start: the token for a
/// Latin-1 character `c` is `CHAR_TOKENS + c`. A mock token lands in that
/// range about once in 17 million.
pub const CHAR_TOKENS: TokenId = 0xFFE0_0000;

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
/// is — by one token of generation prompt. Like the real Qwen template, it
/// then refuses a conversation with no user message.
#[derive(Default)]
pub struct Script {
    decode: HashMap<TokenId, &'static str>,
    boundaries: bool,
    /// What a token `decode` does not name reads as.
    unnamed: &'static str,
    /// Text encodes one [`CHAR_TOKENS`] token per character, and those
    /// tokens decode back: what the loop guard's nudges and tags need.
    chars: bool,
    pub seen: Mutex<Vec<Rendered>>,
}

impl Script {
    /// Tokens decode to `decode`'s text, and to `?` otherwise.
    pub fn new(decode: HashMap<TokenId, &'static str>) -> Arc<Self> {
        Arc::new(Self { decode, unnamed: "?", ..Self::default() })
    }

    /// Every token reads as nothing: a request generates without a single
    /// text event, however long it runs.
    pub fn silent() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Tokens decode to `decode`'s text, and to nothing otherwise: a request
    /// the script does not name generates silently, however long it runs,
    /// while one it does name still streams text.
    pub fn quiet_except(decode: HashMap<TokenId, &'static str>) -> Arc<Self> {
        Arc::new(Self { decode, ..Self::default() })
    }

    /// [`Script::new`], reporting the prompt's structure.
    pub fn with_boundaries(decode: HashMap<TokenId, &'static str>) -> Arc<Self> {
        Arc::new(Self { decode, boundaries: true, unnamed: "?", ..Self::default() })
    }

    /// [`Script::quiet_except`], with a tokenizer for Latin-1 text: one
    /// [`CHAR_TOKENS`] token per character.
    pub fn encoding(decode: HashMap<TokenId, &'static str>) -> Arc<Self> {
        Arc::new(Self { decode, chars: true, ..Self::default() })
    }

    /// [`Script::encoding`], every token it does not name reading as
    /// `unnamed`: what a loop guard continuation, whose fresh seed draws
    /// tokens no script names, writes.
    pub fn encoding_or(decode: HashMap<TokenId, &'static str>, unnamed: &'static str) -> Arc<Self> {
        Arc::new(Self { decode, chars: true, unnamed, ..Self::default() })
    }

    /// What `token` reads as.
    fn piece(&self, token: TokenId) -> String {
        if let Some(text) = self.decode.get(&token) {
            return (*text).to_owned();
        }
        match token.checked_sub(CHAR_TOKENS).filter(|&c| self.chars && c <= 0xFF).and_then(char::from_u32) {
            Some(c) => c.to_string(),
            None => self.unnamed.to_owned(),
        }
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
        if self.0.boundaries && !messages.iter().any(|m| m.role == "user") {
            return Err(TemplateRejection {
                code: "render_failed",
                message: "render chat template: invalid operation: No user query found in messages. (in chat_template.jinja:100)".into(),
            });
        }
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
        tokens.iter().map(|&t| self.0.piece(t)).collect()
    }

    fn encode_literal(&self, text: &str) -> Option<Vec<TokenId>> {
        if !self.0.chars {
            return None;
        }
        text.chars().map(|c| (u32::from(c) <= 0xFF).then_some(CHAR_TOKENS + u32::from(c))).collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(Decoder(Arc::clone(&self.0)))
    }
    // `decoder_starts_in_reasoning` stays the trait default: a request with
    // thinking on starts in the reasoning channel, as a real template's does.
}

struct Decoder(Arc<Script>);

impl TokenDecoder for Decoder {
    fn push(&mut self, token: TokenId) -> String {
        self.0.piece(token)
    }
    fn finish(&mut self) -> String {
        String::new()
    }
}

/// A server over `compute` and `script`, with a fixed wall clock and
/// response ids counted from 0.
pub fn server(script: &Arc<Script>, config: SchedulerConfig, compute: Arc<dyn Compute>) -> Server {
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..config }, compute);
    Server::new(Engine::new(Box::new(scheduler)), Box::new(Handle(Arc::clone(script))))
        .with_request_timeout(Duration::from_secs(5))
        .with_wall_clock(Arc::new(FixedClock::new(CREATED_AT_MS)))
        .with_counted_response_ids()
        .with_seedless_seed(0)
}

// ── the loop guard (ADR 0052) ────────────────────────────────────────────

/// `request`'s tokens on `mock`, from the first, read as `text` cut into
/// pieces of 8 characters: what a [`Script`] decodes them to.
pub fn scripted_text(mock: &MockCompute, request: RequestId, text: &str) -> HashMap<TokenId, &'static str> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(8)
        .enumerate()
        .map(|(step, piece)| {
            let piece: &'static str = Box::leak(piece.iter().collect::<String>().into_boxed_str());
            (mock.token_for(request, step as u32), piece)
        })
        .collect()
}

#[path = "loop_text.rs"]
mod loop_text;
// Not every test binary that includes this file reasons.
#[allow(unused_imports)]
pub use loop_text::{looping, prose};

/// Poll `metrics`' exposition until `line` is in it.
pub async fn metric_reaches(metrics: &ignis_server::metrics::Metrics, line: &str) {
    let found = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if metrics.render().contains(&format!("\n{line}\n")) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(found.is_ok(), "no `{line}` in:\n{}", metrics.render());
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

/// Send one frame as written: a JSON event's text, its key order kept.
pub async fn send_text(socket: &mut Socket, frame: String) {
    socket.send(Message::Text(frame.into())).await.expect("send");
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
