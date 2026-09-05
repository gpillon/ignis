//! The capture proxy (`record` subcommand): records a live agent session as
//! a bench trace.
//!
//! The gate-run (spec 03) needs a stable capture tool: a live session of
//! one main agent + ~10 subagents is recorded into a bench trace that the
//! replay harness re-sends. The capture proxy is a *transparent* OpenAI
//! endpoint in front of the target engine: it accepts chat-completions from
//! the agent client, records each request as a `TraceLine` (`id`, `class`,
//! `t_arrive_ms` — the offset from the session's first request, `prompt` —
//! the actual request content, `max_tokens`, `stream`), forwards the
//! request byte-for-byte to the target engine, and pipes the response back
//! (an SSE stream is streamed, not buffered).
//!
//! The recorded file is the v1 *load trace* — the `TraceLine` shape in
//! `trace.rs`, the same shape `replay` consumes. Lines are appended +
//! flushed on arrival, so the file is always valid JSONL (a proxy that
//! dies mid-session loses nothing); `POST /v1/session/end` finalizes the
//! session (the summary) and stops the proxy.
//!
//! The class policy (`ClassPolicy`) decides which recorded request is the
//! main agent: `first-is-main` (the first request of the session is the
//! main agent; the rest are subagents) or `marker` (the client tags each
//! request with the [`CLASS_HEADER`] header — "main" / "sub"). The trace
//! invariant — at most 1 main request (the replay driver rejects more) —
//! is enforced at capture: a second and later `main` is demoted to `sub`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use reqwest::Client;
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crate::trace::{RequestClass, TraceLine};

/// The client header that tags a request's class (the `marker` class
/// policy): "main" or "sub".
pub const CLASS_HEADER: &str = "x-ignis-class";

/// The class policy of a capture session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ClassPolicy {
    /// The first request of the session is the main agent; the rest are
    /// subagents.
    #[default]
    FirstIsMain,
    /// The client tags each request with the [`CLASS_HEADER`] header
    /// ("main" / "sub"); a missing or unrecognized marker reads as `sub`.
    Marker,
}

impl ClassPolicy {
    /// Parse a `--class` value.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "first-is-main" => Ok(Self::FirstIsMain),
            "marker" => Ok(Self::Marker),
            other => Err(format!(
                "unknown class policy {other:?} (expected `first-is-main` or `marker`)"
            )),
        }
    }

    /// The `--class` value (for summaries / logs).
    pub fn name(self) -> &'static str {
        match self {
            Self::FirstIsMain => "first-is-main",
            Self::Marker => "marker",
        }
    }
}

/// A capture session's configuration.
#[derive(Debug, Clone)]
pub struct RecordConfig {
    /// The proxy's bind address (`"host:port"`; a bare port means
    /// `127.0.0.1:<port>`, and a full `http://host:port` URL is accepted).
    pub listen: String,
    /// The target engine's base URL (e.g. `http://127.0.0.1:8080`).
    pub target: String,
    /// Where the trace is written (JSONL, one `TraceLine` per line).
    pub out: PathBuf,
    /// The class policy (default: `first-is-main`).
    pub class_policy: ClassPolicy,
}

/// A finalized capture session (the `POST /v1/session/end` payload and the
/// CLI's end-of-run summary).
#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
    /// The number of recorded requests.
    pub requests: usize,
    /// The `main`-class requests (at most 1).
    pub main: usize,
    /// The `sub`-class requests.
    pub sub: usize,
    /// The session's span: the last arrival minus the first, in ms (0 with
    /// fewer than 2 requests).
    pub duration_ms: u64,
    /// The trace file (`None` when no request was recorded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    /// The class policy in effect.
    pub class_policy: String,
}

/// A capture session's shared state: the trace-line assignment (id, class,
/// arrival offset) and the on-disk JSONL file. Cheap to clone (an axum
/// state); everything mutable lives behind one lock (a line append is a
/// microseconds-long critical section).
#[derive(Clone)]
pub struct Capture {
    inner: Arc<CaptureInner>,
}

struct CaptureInner {
    out: PathBuf,
    policy: ClassPolicy,
    state: Mutex<SessionState>,
}

#[derive(Default)]
struct SessionState {
    /// The trace file (opened lazily on the first recorded line; each line
    /// is flushed on arrival, so the file is always valid JSONL).
    file: Option<BufWriter<File>>,
    /// The per-request sequence (ids: `req-001`, `req-002`, …).
    seq: usize,
    /// The number of recorded requests.
    total: usize,
    /// The requests recorded as `main` (at most 1 — a second is demoted to
    /// `sub`).
    mains: usize,
    /// The session's first request (the `t_arrive_ms` reference).
    first: Option<Instant>,
    /// The session's last request (the summary's `duration_ms`).
    last: Option<Instant>,
}

impl Capture {
    /// A capture session writing to `out` under `policy`. The parent
    /// directory of `out` is created when missing (an empty `out` is an
    /// error).
    pub fn new(out: PathBuf, policy: ClassPolicy) -> Result<Self, String> {
        if out.as_os_str().is_empty() {
            return Err("the trace path is empty".into());
        }
        // Create the parent directory when there is one (a bare file name
        // has an empty / absent parent — nothing to create).
        if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create the trace directory {}: {e}", parent.display()))?;
        }
        Ok(Self {
            inner: Arc::new(CaptureInner {
                out,
                policy,
                state: Mutex::new(SessionState::default()),
            }),
        })
    }

    /// Record one request (the client's actual body) as a trace line, and
    /// append + flush it to the trace file.
    ///
    /// `now` is the arrival instant (the caller passes `Instant::now()`);
    /// the line's `t_arrive_ms` is its offset from the session's first
    /// request (which is 0). `marker` is the client's class marker (the
    /// [`CLASS_HEADER`] value, if any), read under the `marker` policy.
    /// Returns the recorded line.
    pub fn record(&self, raw_body: &str, marker: Option<&str>, now: Instant) -> Result<TraceLine, String> {
        let mut s = self.inner.state.lock().unwrap();
        // Lazy open: the file is created by the first recorded request (a
        // new session is a new file — a previous run's file is truncated).
        if s.file.is_none() {
            let f = File::create(&self.inner.out)
                .map_err(|e| format!("create {}: {e}", self.inner.out.display()))?;
            s.file = Some(BufWriter::new(f));
        }
        let is_first = s.first.is_none();
        s.first.get_or_insert(now);
        s.last = Some(now);
        let seq = s.seq + 1;
        s.seq = seq;
        let t_arrive_ms = elapsed_ms(s.first.expect("just set"), now);

        // The class (per policy) — and the trace invariant (at most 1 main
        // request; the `replay` driver rejects more): a second and later
        // `main` is demoted to `sub` (the load shape is "1 main + N
        // subagents").
        let mut class = match self.inner.policy {
            ClassPolicy::FirstIsMain => {
                if is_first {
                    RequestClass::Main
                } else {
                    RequestClass::Sub
                }
            }
            ClassPolicy::Marker => match marker {
                Some("main") => RequestClass::Main,
                _ => RequestClass::Sub,
            },
        };
        if class == RequestClass::Main {
            if s.mains >= 1 {
                eprintln!("ignis-bench record: a second `main` request was recorded; the trace allows at most 1 main (the `replay` driver rejects more) — demoted to `sub`");
                class = RequestClass::Sub;
            } else {
                s.mains += 1;
            }
        }

        // The request's own fields: `max_tokens` (or OpenAI's
        // `max_completion_tokens` alias; the trace's default when absent)
        // and `stream` (a missing flag reads as non-streaming). Everything
        // else of the request is the `prompt` (the actual content — the
        // engine sees the same input the reference saw).
        let value: serde_json::Value =
            serde_json::from_str(raw_body).unwrap_or(serde_json::Value::Null);
        let max_tokens = value
            .get("max_tokens")
            .or_else(|| value.get("max_completion_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1024)
            .min(u32::MAX as u64) as u32;
        let stream = value.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

        let line = TraceLine {
            id: format!("req-{seq:03}"),
            class,
            t_arrive_ms,
            prompt: raw_body.to_string(),
            max_tokens,
            stream,
        };
        s.total += 1;

        let json = serde_json::to_string(&line).map_err(|e| format!("serialize the trace line: {e}"))?;
        let w = s.file.as_mut().expect("file opened above");
        w.write_all(json.as_bytes())
            .and_then(|_| w.write_all(b"\n"))
            .and_then(|_| w.flush())
            .map_err(|e| format!("write {}: {e}", self.inner.out.display()))?;
        Ok(line)
    }

    /// Finalize the session: flush the trace file and build the summary
    /// (idempotent — the file is valid either way).
    pub fn finalize(&self) -> SessionSummary {
        let mut s = self.inner.state.lock().unwrap();
        if let Some(w) = s.file.as_mut() {
            let _ = w.flush();
        }
        let duration_ms = match (s.first, s.last) {
            (Some(first), Some(last)) => elapsed_ms(first, last),
            _ => 0,
        };
        SessionSummary {
            requests: s.total,
            main: s.mains,
            sub: s.total - s.mains,
            duration_ms,
            file: if s.total > 0 { Some(self.inner.out.clone()) } else { None },
            class_policy: self.inner.policy.name().to_string(),
        }
    }
}

/// Milliseconds between two instants (0 when the clock went backwards —
/// `Instant` is monotonic, so this only guards the arithmetic).
fn elapsed_ms(from: Instant, to: Instant) -> u64 {
    to.checked_duration_since(from)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The capture proxy: the axum app that records + forwards.
pub struct RecordServer {
    cfg: RecordConfig,
    capture: Capture,
    client: Client,
}

impl RecordServer {
    /// Build the session's proxy (no runtime yet; `bind` / `serve` run in a
    /// tokio context).
    pub fn new(cfg: RecordConfig) -> Result<Self, String> {
        let capture = Capture::new(cfg.out.clone(), cfg.class_policy)?;
        let target = cfg.target.trim_end_matches('/').to_string();
        if target.is_empty() {
            return Err("the target URL is empty".into());
        }
        Ok(Self {
            cfg: RecordConfig {
                target,
                ..cfg
            },
            capture,
            client: Client::new(),
        })
    }

    /// The capture session's state (the trace file + counters).
    pub fn capture(&self) -> &Capture {
        &self.capture
    }

    /// The target engine's base URL (normalized: no trailing `/`).
    pub fn target(&self) -> &str {
        &self.cfg.target
    }

    /// Bind the proxy's listener. Returns the proxy's base URL (the
    /// actually-bound address — an ephemeral port resolves to the assigned
    /// one) and the listener to serve on.
    pub async fn bind(&self) -> Result<(String, TcpListener), String> {
        let addr = normalize_listen(&self.cfg.listen)?;
        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| format!("bind {addr}: {e}"))?;
        let local = listener
            .local_addr()
            .map_err(|e| format!("the bound address: {e}"))?;
        Ok((format!("http://{local}"), listener))
    }

    /// Serve the session: run the proxy until the session ends —
    /// `POST /v1/session/end` (finalizes the trace) or the process dies
    /// (the trace is already complete: lines are flushed on arrival).
    /// Returns the session summary.
    pub async fn serve(self, listener: TcpListener) -> Result<SessionSummary, String> {
        let (stop_tx, mut stop_rx) = mpsc::channel(1);
        let state = ProxyState {
            capture: self.capture.clone(),
            client: self.client.clone(),
            target: self.cfg.target.clone(),
            stop: stop_tx,
        };
        let app = Router::new()
            .route("/v1/chat/completions", post(completions))
            .route("/v1/models", get(models))
            .route("/v1/session/end", post(session_end))
            // A chat-completions body carries the whole conversation, but
            // the `Bytes` extractor's default body limit is 2 MiB (too
            // small for a coding-agent context): lift it to 32 MiB (still
            // bounded).
            .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
            .with_state(state);
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                // The session is over when `session_end` signals (or the
                // signal's senders are all dropped).
                stop_rx.recv().await;
            })
            .await
            .map_err(|e| format!("the capture proxy: {e}"))?;
        Ok(self.capture.finalize())
    }
}

/// A bind address: `"host:port"`, a bare port (`127.0.0.1:<port>`), or a
/// full `http://host:port` URL.
fn normalize_listen(listen: &str) -> Result<String, String> {
    let v = listen.trim();
    let v = v
        .strip_prefix("http://")
        .or_else(|| v.strip_prefix("https://"))
        .unwrap_or(v);
    if v.contains(':') {
        Ok(v.to_string())
    } else {
        Ok(format!("127.0.0.1:{v}"))
    }
}

/// The proxy's shared state (behind `axum::extract::State`): the capture
/// session, the forwarding client, and the session-end stop channel.
#[derive(Clone)]
struct ProxyState {
    capture: Capture,
    client: Client,
    target: String,
    stop: mpsc::Sender<()>,
}

impl ProxyState {
    /// Forward a chat-completions request to the target engine and pipe
    /// the response back (an SSE stream, byte-for-byte).
    async fn forward_completions(&self, headers: &HeaderMap, body: Bytes) -> Response {
        let url = format!("{}/v1/chat/completions", self.target);
        let mut req = self.client.post(&url);
        // The client's headers (auth, etc.) — minus the ones the proxy
        // owns: `host` (the target is a different host), `content-length`
        // (reqwest computes it for the body), and the connection framing.
        for (name, value) in headers {
            let name = name.as_str();
            if name.eq_ignore_ascii_case("host")
                || name.eq_ignore_ascii_case("content-length")
                || name.eq_ignore_ascii_case("connection")
                || name.eq_ignore_ascii_case("transfer-encoding")
            {
                continue;
            }
            req = req.header(name, value.to_str().unwrap_or_default());
        }
        if !headers.contains_key("content-type") {
            req = req.header("content-type", "application/json");
        }
        match req.body(body).send().await {
            Ok(resp) => pipe_response(resp),
            Err(e) => {
                eprintln!("ignis-bench record: target {url}: {e}");
                target_error(&url, e)
            }
        }
    }
}

/// `POST /v1/chat/completions` — record the request (a `TraceLine`), then
/// forward the raw body to the target engine and pipe the response back.
async fn completions(State(st): State<ProxyState>, headers: HeaderMap, body: Bytes) -> Response {
    // The actual request content (the trace line's `prompt`).
    let raw = String::from_utf8_lossy(&body).into_owned();
    let marker = headers.get(CLASS_HEADER).and_then(|v| v.to_str().ok());
    let line = match st.capture.record(&raw, marker, Instant::now()) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("ignis-bench record: {e}");
            return record_error(&e);
        }
    };
    eprintln!(
        "  recorded {} [{}] +{} ms",
        line.id,
        class_name(line.class),
        line.t_arrive_ms
    );
    st.forward_completions(&headers, body).await
}

/// `GET /v1/models` — a pass-through to the target (an agent client probes
/// the model list; nothing is recorded — only chat-completions are load
/// requests).
async fn models(State(st): State<ProxyState>) -> Response {
    let url = format!("{}/v1/models", st.target);
    match st.client.get(&url).send().await {
        Ok(resp) => pipe_response(resp),
        Err(e) => {
            eprintln!("ignis-bench record: target {url}: {e}");
            target_error(&url, e)
        }
    }
}

/// `POST /v1/session/end` — the session is over: finalize the trace
/// (flush + the summary) and stop the proxy (the in-flight request is this
/// one; once it is sent, the server drains and exits).
async fn session_end(State(st): State<ProxyState>) -> impl IntoResponse {
    let summary = st.capture.finalize();
    let _ = st.stop.send(()).await;
    Json(summary)
}

/// Pipe the target's response back: the status + the content headers, and
/// the body streamed byte-for-byte (`Body::from_stream` — an SSE token is
/// visible to the client as soon as the engine emits it).
fn pipe_response(resp: reqwest::Response) -> Response {
    let status = resp.status();
    // The filtered header set: clone the target's headers (owned — no
    // borrow of `resp`), then drop the framing headers (they are
    // recomputed for the proxy's own response — the body is chunked /
    // close-framed, not content-length-framed).
    let mut headers = resp.headers().clone();
    for key in ["content-length", "transfer-encoding", "connection"] {
        headers.remove(key);
    }
    let body = Body::from_stream(resp.bytes_stream());
    // `http::response::Builder` has no `headers` setter: build the
    // response first, then install the filtered header set (the framing
    // headers were stripped above, so the body is chunked / close-framed).
    let mut response = match Response::builder().status(status).body(body) {
        Ok(response) => response,
        Err(e) => {
            eprintln!("ignis-bench record: invalid response from the target: {e}");
            return record_error("invalid response from the target");
        }
    };
    *response.headers_mut() = headers;
    response
}

/// A 502 in the engine's error shape (the target is unreachable).
fn target_error(url: &str, e: reqwest::Error) -> Response {
    (
        StatusCode::BAD_GATEWAY,
        Json(serde_json::json!({
            "error": {
                "message": format!("target {url}: {e}"),
                "type": "target_unreachable",
                "code": "target_unreachable"
            }
        })),
    )
        .into_response()
}

/// A 500 in the engine's error shape (the proxy itself failed — e.g. the
/// trace file cannot be written).
fn record_error(message: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "record_error",
                "code": "record_error"
            }
        })),
    )
        .into_response()
}

/// A class's name (logs + summaries: "main" / "sub").
fn class_name(class: RequestClass) -> &'static str {
    match class {
        RequestClass::Main => "main",
        RequestClass::Sub => "sub",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static NEXT_TMP: AtomicU64 = AtomicU64::new(0);

    /// A unique temp JSONL path (the in-process tests share one pid).
    fn temp_out() -> PathBuf {
        let n = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ignis-bench-record-{pid}-{n}.jsonl", pid = std::process::id()))
    }

    /// A real chat-completions body (the OpenAI wire shape the proxy sees).
    const MAIN_BODY: &str = r#"{"model":"qwen","messages":[{"role":"user","content":"Hello from the main agent"}],"max_tokens":256,"stream":true}"#;
    const SUB_BODY: &str = r#"{"model":"qwen","messages":[{"role":"user","content":"Sub one"}],"max_tokens":64}"#;

    #[test]
    fn a_session_records_lines_in_the_trace_shape() {
        let out = temp_out();
        let cap = Capture::new(out.clone(), ClassPolicy::FirstIsMain).expect("capture");
        let t0 = Instant::now();
        let l1 = cap.record(MAIN_BODY, None, t0).expect("first request");
        let l2 = cap
            .record(SUB_BODY, None, t0 + Duration::from_millis(40))
            .expect("second request");
        let l3 = cap
            .record(SUB_BODY, None, t0 + Duration::from_millis(40))
            .expect("third request");

        // The ids + the classes (the `first-is-main` policy).
        assert_eq!(l1.id, "req-001");
        assert_eq!(l2.id, "req-002");
        assert_eq!(l3.id, "req-003");
        assert_eq!(l1.class, RequestClass::Main);
        assert_eq!(l2.class, RequestClass::Sub);
        assert_eq!(l3.class, RequestClass::Sub);

        // The arrival offsets (the session's first request is 0; ties are
        // stable — both measure the same instant).
        assert_eq!(l1.t_arrive_ms, 0);
        assert_eq!(l2.t_arrive_ms, 40);
        assert_eq!(l3.t_arrive_ms, 40);

        // The actual request content + the request's own fields.
        assert!(l1.prompt.contains("Hello from the main agent"));
        assert_eq!(l1.max_tokens, 256);
        assert!(l1.stream);
        assert_eq!(l2.max_tokens, 64);
        assert!(!l2.stream, "a missing `stream` reads as non-streaming");

        // The file is valid JSONL and parses as a `Trace` (the shape
        // `replay` consumes).
        let text = std::fs::read_to_string(&out).expect("the trace file is written");
        let trace = crate::trace::Trace::from_jsonl(&text).expect("the recorded trace parses");
        assert_eq!(trace.len(), 3);

        let summary = cap.finalize();
        assert_eq!(summary.requests, 3);
        assert_eq!(summary.main, 1);
        assert_eq!(summary.sub, 2);
        assert_eq!(summary.duration_ms, 40);
        assert_eq!(summary.file, Some(out.clone()));
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn the_marker_policy_reads_the_client_marker() {
        let out = temp_out();
        let cap = Capture::new(out.clone(), ClassPolicy::Marker).expect("capture");
        let t0 = Instant::now();
        let l1 = cap.record(SUB_BODY, Some("main"), t0).expect("main marker");
        let l2 = cap
            .record(SUB_BODY, Some("sub"), t0 + Duration::from_millis(10))
            .expect("sub marker");
        let l3 = cap
            .record(SUB_BODY, Some("nope"), t0 + Duration::from_millis(20))
            .expect("unknown marker");
        let l4 = cap
            .record(SUB_BODY, None, t0 + Duration::from_millis(20))
            .expect("missing marker");
        assert_eq!(l1.class, RequestClass::Main);
        assert_eq!(l2.class, RequestClass::Sub);
        assert_eq!(l3.class, RequestClass::Sub);
        assert_eq!(l4.class, RequestClass::Sub);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn a_second_main_is_demoted_to_sub() {
        // The trace invariant (at most 1 main — the `replay` driver
        // rejects more): a second `main` marker is demoted, and the file
        // stays loadable.
        let out = temp_out();
        let cap = Capture::new(out.clone(), ClassPolicy::Marker).expect("capture");
        let t0 = Instant::now();
        let l1 = cap.record(SUB_BODY, Some("main"), t0).expect("first main");
        let l2 = cap
            .record(SUB_BODY, Some("main"), t0 + Duration::from_millis(10))
            .expect("second main");
        assert_eq!(l1.class, RequestClass::Main);
        assert_eq!(l2.class, RequestClass::Sub);
        let summary = cap.finalize();
        assert_eq!(summary.main, 1);
        let text = std::fs::read_to_string(&out).expect("the trace file is written");
        crate::trace::Trace::from_jsonl(&text).expect("the trace stays valid");
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn an_empty_session_summarizes_to_zero() {
        let out = temp_out();
        let cap = Capture::new(out.clone(), ClassPolicy::FirstIsMain).expect("capture");
        let summary = cap.finalize();
        assert_eq!(summary.requests, 0);
        assert_eq!(summary.main, 0);
        assert_eq!(summary.sub, 0);
        assert_eq!(summary.duration_ms, 0);
        assert_eq!(summary.file, None);
        assert!(!out.exists(), "no requests, no file");
    }

    #[test]
    fn the_class_policy_parses_its_cli_values() {
        assert_eq!(ClassPolicy::parse("first-is-main"), Ok(ClassPolicy::FirstIsMain));
        assert_eq!(ClassPolicy::parse("marker"), Ok(ClassPolicy::Marker));
        assert!(ClassPolicy::parse("bogus").is_err());
    }

    #[test]
    fn the_listen_address_normalizes() {
        assert_eq!(normalize_listen("127.0.0.1:8090"), Ok("127.0.0.1:8090".into()));
        assert_eq!(
            normalize_listen("http://127.0.0.1:8090"),
            Ok("127.0.0.1:8090".into())
        );
        assert_eq!(normalize_listen("8090"), Ok("127.0.0.1:8090".into()));
    }
}