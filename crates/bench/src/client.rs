//! The replay client: drives the engine endpoint and collects per-request
//! metrics.
//!
//! The harness talks to an OpenAI-compatible engine over HTTP. The I/O is
//! isolated behind the `Endpoint` trait so the replay driver + metrics logic
//! is testable against a `MockEndpoint` with **no** running server. The real
//! `HttpEndpoint` (a `reqwest` blocking client) drives the running
//! `ignis-server` endpoint (`POST /v1/chat/completions` — streaming SSE for
//! per-token timing + non-streaming — and `GET /v1/models`) and measures the
//! per-request timing (see `docs/specs/bench/01-trace-replay.md`).
//!
//! The replay driver runs a bounded-concurrency worker pool: jobs (requests
//! with their arrival offsets) flow through an mpsc channel into
//! `max_concurrency` worker threads, which model the "1 main + N subagents"
//! concurrent load. Arrival offsets are honored (scaled by `ReplayConfig`),
//! so the scheduler sees a realistic concurrency profile.

use std::collections::VecDeque;
use std::io::BufRead;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest::blocking::{Client, Response};

use crate::metrics::RequestMetrics;
use crate::trace::{RequestClass, Trace};

/// A request the replay driver can send to an endpoint.
#[derive(Debug, Clone)]
pub struct Request {
    /// Stable request id (matches the trace line and the reference run).
    pub id: String,
    /// Class: main agent vs subagent (per-class gate, ADR 0007).
    pub class: RequestClass,
    /// The prompt to send.
    pub prompt: String,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Whether the request was streaming (affects ttft).
    pub stream: bool,
    /// `stream_options.include_usage` — ask a streaming response for the
    /// trailing usage chunk. The TTFT instrument (`ttft.rs`) needs it: the
    /// usage figures are how a sample proves its prefix was cold (the
    /// engine's own computed-prefill-token count). Replay traffic leaves it
    /// `false` — one fewer chunk to parse per request.
    pub include_usage: bool,
    /// `enable_thinking` to send on the request (GitHub #68), or `None` to
    /// omit the field (the server's configured default applies). Only the
    /// canary oracle recorder/comparer set this explicitly — see
    /// `oracle::record`'s doc comment for why the G1 comparison needs
    /// thinking disabled.
    pub enable_thinking: Option<bool>,
    /// Images to send after [`Request::prompt`], each a `data:` URI or an
    /// HTTP(S) URL (GitHub #181: the multimodal TTFT cell). Empty sends the
    /// plain-string content every text request always sent; otherwise the
    /// user message becomes OpenAI content parts, the prompt's text part
    /// first and then one `image_url` part per image, in order.
    pub images: Vec<String>,
}

/// Why a request's token stream ended.
///
/// The G3 ITL cell's decode lanes are meant to end when the harness closes
/// them at the measurement boundary. Anything else means the fixture, not
/// the measurement, decided how long a lane lived, and #110's own run had
/// one leg end that way without the record showing it (GitHub #114).
/// Reading this costs one field of a chunk this client already parses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The engine named a reason in the response's `finish_reason`
    /// (`"stop"`, `"length"`, ...), carried verbatim rather than mapped
    /// here: what a reason *means* to a measurement is the measurement's
    /// business, not the transport's.
    Engine(String),
    /// The observer returned `false`, so this client stopped reading and
    /// dropped the body, cancelling the generation server-side. The
    /// engine's own reason, if it would have had one, was never sent.
    Cancelled,
}

/// The raw outcome of a single request completion.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// Time to first token (ms), measured **per chunk**: the arrival of the
    /// first chunk carrying text. The first generated token of *either*
    /// channel — a turn that thinks before it answers spent its prefill
    /// exactly once (GitHub #137).
    pub ttft_ms: f64,
    /// Wall-clock duration of the whole request (ms).
    pub total_ms: f64,
    /// Number of tokens generated, in **tokens**: the engine's own
    /// `usage.completion_tokens` when the stream carried it (a request
    /// with [`Request::include_usage`]) and it ran to its end, else the
    /// **chunk** count, which undercounts an engine that packs a verify
    /// round's run into one chunk — and is all a cancelled stream has, since
    /// the usage chunk comes last. Both channels together — the thinking
    /// channel is decode work like any other, and a throughput cell that
    /// dropped it would read zero for a turn that answers entirely inside
    /// `<think>` (GitHub #137). [`Outcome::reasoning_tokens`] carries the
    /// split for a caller that wants the two rates apart.
    pub n_tokens: u32,
    /// The generated text of the **content** channel. Correctness checks
    /// (the oracle comparison, needle retrieval) read this and only this:
    /// scratch work is not an answer.
    pub output: String,
    /// The generated text of the **reasoning** channel
    /// (`delta.reasoning_content` / `message.reasoning_content`), empty when
    /// the turn did not think or thinking was disabled.
    pub reasoning_output: String,
    /// How many **chunks** carried text on the reasoning channel (the usage
    /// figure does not split channels), so it stays in the unit of
    /// [`Outcome::token_times_ms`] even when usage replaced
    /// [`Outcome::n_tokens`]. It is a count of tokens, not a partition of them: a chunk
    /// that somehow carried both channels at once counts once in each, so
    /// `n_tokens - reasoning_tokens` is a lower bound on the answer tokens
    /// rather than their exact number. The server never sends such a chunk
    /// (GitHub #68).
    ///
    /// `None` means this transport cannot tell the channels apart: a
    /// non-streaming response reports one `completion_tokens` figure and no
    /// per-channel split, which is not the same as a reported zero.
    pub reasoning_tokens: Option<u32>,
    /// `usage.prompt_tokens` as the engine reported it, when it reported
    /// any (a streaming response reports it only with
    /// [`Request::include_usage`]).
    pub prompt_tokens: Option<u32>,
    /// `usage.prompt_tokens_details.cached_tokens` — the prompt tokens the
    /// engine served from a prefix cache instead of computing. Absent means
    /// the engine does not report the field at all, which is not the same
    /// as a reported zero; see [`Outcome::computed_prefill_tokens`].
    pub cached_prompt_tokens: Option<u32>,
    /// The arrival time of every generated **chunk**, in ms since the request
    /// was sent (one entry per chunk, so its length is the chunk count, not
    /// [`Outcome::n_tokens`] when usage replaced it; streaming only; empty for a non-streaming response). This is
    /// what the G3 inter-token-latency cell reads: a decode lane's inter-
    /// token intervals are the consecutive differences of this series
    /// ([`crate::g3`]), sampled continuously rather than reduced to a
    /// single ttft/total pair. Both channels land here, in arrival order —
    /// the decode cadence is the decode cadence whatever the token says.
    pub token_times_ms: Vec<f64>,
    /// Why the stream ended, when this transport could tell (GitHub #114).
    /// `None` for a response that named no reason and was not cancelled,
    /// and for transports that report none.
    pub finish_reason: Option<FinishReason>,
}

impl Outcome {
    /// The prompt tokens the engine actually *computed* this request: the
    /// reported prompt tokens minus whatever it served from a prefix cache.
    ///
    /// This is the one reading the cold-prefix rule is judged on (ADR 0015),
    /// and both engines are read the same way — the OpenAI usage shape both
    /// speak. `None` means the engine reported no usage at all, which the
    /// TTFT instrument treats as an unverifiable (void) sample rather than
    /// as a cold one: a cold-prefix claim has to be evidence.
    pub fn computed_prefill_tokens(&self) -> Option<u32> {
        self.prompt_tokens
            .map(|prompt| prompt.saturating_sub(self.cached_prompt_tokens.unwrap_or(0)))
    }
}

/// How much of a request's life went to waiting for the engine to admit it
/// (GitHub #329 item 14), as opposed to being served: the `engine_full`
/// refusals it was resent after, and the time spent waiting between them.
/// An [`Outcome`]'s own timings start at the attempt that was admitted, so
/// a report can tell queueing from service.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Queueing {
    /// How many times the request was refused with `engine_full` and resent.
    pub retries: u32,
    /// The wall time slept waiting to resend, summed over the retries (ms).
    /// It leaves out the refused round trips themselves (each refusal's
    /// request and response), so it is the waiting the harness chose, not
    /// the whole time lost to the queue.
    pub wait_ms: f64,
}

/// Opt-in retry of a `503` `engine_full` refusal (`--retry-engine-full`).
/// The HTTP front end never queues a request it cannot admit: the client's
/// retry, after the `Retry-After` it was told, is the admission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryPolicy {
    /// The most times one request is resent after a refusal.
    pub max_retries: u32,
    /// The longest a single wait may be: the `Retry-After` header's seconds
    /// are bounded to this, so a stale estimate cannot park the run. At most
    /// [`MAX_RETRY_WAIT`], the server's own `Retry-After` range.
    pub max_wait: Duration,
    /// The most time one request may spend waiting, summed over its
    /// retries: a request that long-held lanes (a `max_tokens` of tens of
    /// thousands runs for minutes) keep refusing is given up on at this
    /// budget, not after a count of short waits. The last wait is cut to
    /// what is left.
    pub total_wait: Duration,
}

/// The longest `--retry-max-wait` accepted: the server's `Retry-After` is
/// 1 to 60 seconds (`server/src/api.rs`), so a longer cap waits for nothing.
pub const MAX_RETRY_WAIT: Duration = Duration::from_secs(60);

impl Default for RetryPolicy {
    fn default() -> Self {
        // The budget, not the count, is what ends a retry by default.
        Self { max_retries: 100, max_wait: Duration::from_secs(30), total_wait: Duration::from_secs(600) }
    }
}

impl RetryPolicy {
    /// How long to wait before the next resend after `waited_ms` of waiting
    /// so far, or `None` when the budget is spent.
    fn next_wait(&self, retry_after: Option<&str>, waited_ms: f64) -> Option<Duration> {
        let left = self.total_wait.as_secs_f64() * 1000.0 - waited_ms;
        if left <= 0.0 {
            return None;
        }
        Some(retry_wait(retry_after, self.max_wait).min(Duration::from_secs_f64(left / 1000.0)))
    }
}

/// How long to wait before resending a refused request: the response's
/// `Retry-After` (whole seconds; 1 s when it is absent or not a number),
/// bounded to `max_wait`.
fn retry_wait(retry_after: Option<&str>, max_wait: Duration) -> Duration {
    let secs = retry_after.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1);
    Duration::from_secs(secs).min(max_wait)
}

/// Whether a refusal's JSON body names the `engine_full` code (the server's
/// `{"error": {"code": "engine_full"}}` shape).
fn is_engine_full(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.pointer("/error/code").and_then(|c| c.as_str()).map(|c| c == "engine_full"))
        .unwrap_or(false)
}

/// An endpoint the replay driver can send requests to. `Send + Sync` so a
/// single shared endpoint can be used by the driver's worker threads.
pub trait Endpoint: Send + Sync {
    /// Send one request and return its outcome. Returns a human-readable error
    /// string on failure (the driver records a failed request, not a panic).
    fn complete(&self, req: &Request) -> Result<Outcome, String>;

    /// [`Endpoint::complete`], plus the [`Queueing`] the request went
    /// through to be admitted (reported even when it finally failed). An
    /// endpoint that never retries needs no override.
    fn complete_queued(&self, req: &Request) -> (Result<Outcome, String>, Queueing) {
        (self.complete(req), Queueing::default())
    }

    /// Complete a request while reporting each generated-token arrival, with
    /// no way to stop early. A convenience wrapper over
    /// [`Endpoint::complete_observed_while`] — override *that* method, not
    /// this one, so a streaming transport's observers run as chunks arrive
    /// rather than after completion.
    fn complete_observed(
        &self,
        req: &Request,
        observer: &mut dyn FnMut(f64),
    ) -> Result<Outcome, String> {
        self.complete_observed_while(
            req,
            &mut |time_ms| {
                observer(time_ms);
                true
            },
            &[],
        )
    }

    /// Stream a request while `observer` wants more tokens. Returning
    /// `false` stops reading the response immediately; HTTP transports
    /// thereby cancel the server-side generation by dropping the body.
    ///
    /// `suppress_eos` names the token ids a measurement lane must not stop
    /// on (see [`suppress_eos_in`]). It is a *request* the
    /// endpoint may be unable to grant: this default ignores it, and even
    /// the HTTP endpoint can only pass it on, as the one control its engine
    /// is sent ([`HttpEndpoint::eos_control`]) — an engine that honours
    /// neither `ignore_eos` nor `logit_bias` still stops at its own EOS.
    /// A caller that depends on a lane outliving EOS must check the lane's
    /// token count rather than assume the request was honoured.
    fn complete_observed_while(
        &self,
        req: &Request,
        observer: &mut dyn FnMut(f64) -> bool,
        suppress_eos: &[u32],
    ) -> Result<Outcome, String> {
        // No streaming transport here, so there is no request body to carry
        // the suppression onto the wire.
        let _ = suppress_eos;
        let mut outcome = self.complete(req)?;
        for &time_ms in &outcome.token_times_ms {
            if !observer(time_ms) {
                // The caller stopped this stream, so whatever reason the
                // canned outcome carried is not why it ended here (GitHub
                // #114).
                outcome.finish_reason = Some(FinishReason::Cancelled);
                break;
            }
        }
        Ok(outcome)
    }
}

/// A mock endpoint for tests: returns canned outcomes in order and records the
/// requests it received. Thread-safe (all state under a Mutex), so it can be
/// shared by the driver's worker threads.
#[derive(Debug)]
pub struct MockEndpoint {
    /// Canned outcomes consumed in order (one per `complete` call).
    canned: Mutex<VecDeque<Outcome>>,
    /// Every request received so far (for assertions in tests).
    pub received: Mutex<Vec<Request>>,
}

impl MockEndpoint {
    /// Build a mock that serves `canned` outcomes in order. Once the list is
    /// exhausted it falls back to a deterministic per-request outcome
    /// (`fallback_for`).
    pub fn new(canned: Vec<Outcome>) -> Self {
        Self {
            canned: Mutex::new(canned.into()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// A mock with no canned outcomes: every request gets a deterministic
    /// fallback computed from the request (see `fallback_for`).
    pub fn deterministic() -> Self {
        Self {
            canned: Mutex::new(VecDeque::new()),
            received: Mutex::new(Vec::new()),
        }
    }

    /// A deterministic fallback outcome for a request: ttft = 100 ms, and
    /// `max_tokens` tokens at 100 tok/s decode (so total = ttft +
    /// (n_tokens - 1) * 10 ms).
    fn fallback_for(req: &Request) -> Outcome {
        let n = req.max_tokens;
        let ttft = 100.0_f64;
        let decode_ms = if n > 1 { (n as f64 - 1.0) * 10.0 } else { 0.0 };
        // One arrival time per token, `ttft` then 10 ms apart — matching
        // `decode_ms` above so `token_times_ms.last() == total_ms`.
        let token_times_ms = if n == 0 {
            Vec::new()
        } else {
            (0..n).map(|i| ttft + i as f64 * 10.0).collect()
        };
        Outcome {
            ttft_ms: ttft,
            total_ms: ttft + decode_ms,
            n_tokens: n,
            output: format!("[{}]", req.id),
            // The mock never thinks: every token it generates is content.
            reasoning_output: String::new(),
            reasoning_tokens: Some(0),
            // The mock has no tokenizer: it reports the prompt's whitespace
            // word count as its prompt tokens, all of them computed.
            prompt_tokens: Some(req.prompt.split_whitespace().count() as u32),
            cached_prompt_tokens: None,
            token_times_ms,
            // The mock emits exactly `max_tokens` tokens, so the honest
            // reason is the one a real engine would give for that.
            finish_reason: Some(FinishReason::Engine("length".to_string())),
        }
    }
}

impl Default for MockEndpoint {
    fn default() -> Self {
        Self::deterministic()
    }
}

impl Endpoint for MockEndpoint {
    fn complete(&self, req: &Request) -> Result<Outcome, String> {
        self.received.lock().unwrap().push(req.clone());
        if let Some(outcome) = self.canned.lock().unwrap().pop_front() {
            return Ok(outcome);
        }
        Ok(Self::fallback_for(req))
    }
}

/// The per-request deadline the harness gives an engine by default.
///
/// This exists because `reqwest::blocking::Client::new()` does **not** mean
/// "no deadline": the blocking builder defaults `timeout` to **30 seconds**,
/// and that deadline covers reading the response body, not just connecting.
/// A measurement request whose prefill outlives it dies mid-stream with
/// `read SSE: error decoding response body` while the engine is still
/// computing perfectly normally — which is how the G4 needle cell at 131,072
/// tokens came back as a deterministic failure against *both* engines while
/// the 65,536-token cell (a prefill that fits inside 30 s) passed every time
/// (GitHub #138).
///
/// The value is deliberately far above any legitimate measurement request:
/// the 128K needle cell answers in ~60 s on this hardware, and a G3 ITL
/// lane streams thousands of tokens after its own prefill. It is a backstop
/// against a wedged engine, not a pacing knob — [`REQUEST_TIMEOUT_ENV`]
/// raises or removes it for a run that needs to.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// The connect-phase deadline: a dead endpoint must fail on reaching the
/// engine rather than sit out [`DEFAULT_REQUEST_TIMEOUT`]. It is the 30 s
/// the transport already had for the whole request, kept for the one phase
/// that never legitimately takes minutes — so no endpoint that used to be
/// reachable becomes unreachable here; the deadline only stops applying to
/// the part of the request that does the work.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Overrides [`DEFAULT_REQUEST_TIMEOUT`] for a run, in **seconds**. `0`
/// removes the deadline entirely (the harness then waits as long as the
/// engine takes). An unparseable value falls back to the default rather
/// than failing the run — with a line on stderr, since a silently ignored
/// timeout is exactly the class of bug this constant exists for.
pub const REQUEST_TIMEOUT_ENV: &str = "IGNIS_BENCH_REQUEST_TIMEOUT";

/// Read [`REQUEST_TIMEOUT_ENV`]'s value into a client deadline: `None` is
/// "no deadline", which both an absent variable and a `0` can *not* mean at
/// once — an absent variable takes [`DEFAULT_REQUEST_TIMEOUT`], `0` removes
/// the deadline.
fn timeout_from_env(raw: Option<&str>) -> Option<Duration> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Some(DEFAULT_REQUEST_TIMEOUT);
    };
    match raw.parse::<u64>() {
        Ok(0) => None,
        Ok(secs) => Some(Duration::from_secs(secs)),
        Err(_) => {
            eprintln!(
                "warning: {REQUEST_TIMEOUT_ENV}={raw} is not a whole number of seconds; \
                 using the default ({} s)",
                DEFAULT_REQUEST_TIMEOUT.as_secs()
            );
            Some(DEFAULT_REQUEST_TIMEOUT)
        }
    }
}

/// Whether this I/O error is the client giving up on its own deadline
/// rather than the connection actually breaking.
///
/// It cannot be read off the error's kind: reqwest hands a body-read
/// timeout up as `io::ErrorKind::Other` wrapping a `reqwest::Error`, which
/// reads back as the same opaque `error decoding response body` a real
/// transport fault gives. That indistinguishability is half of what made
/// GitHub #138 cost four launches against two engines — so the wrapped
/// error is unwrapped and asked directly.
fn is_client_deadline(err: &std::io::Error) -> bool {
    let mut source: Option<&(dyn std::error::Error + 'static)> =
        err.get_ref().map(|e| e as &(dyn std::error::Error + 'static));
    while let Some(e) = source {
        if e.downcast_ref::<reqwest::Error>().is_some_and(reqwest::Error::is_timeout) {
            return true;
        }
        source = e.source();
    }
    false
}

/// What to say when the harness's own deadline, not the engine, ended a
/// request: name the deadline and how to move it (GitHub #138).
fn deadline_elapsed(method: &str, url: &str, phase: &str) -> String {
    format!(
        "{method} {url}: {phase}: the client's own request deadline elapsed while the engine was \
         still responding — raise it with {REQUEST_TIMEOUT_ENV}=<seconds> (0 removes it)"
    )
}

/// A `send` or a body read that failed: the deadline named when it was the
/// client's own ([`deadline_elapsed`]), the transport's error otherwise.
fn transport_error(method: &str, url: &str, phase: &str, e: &reqwest::Error) -> String {
    if e.is_timeout() {
        deadline_elapsed(method, url, phase)
    } else {
        format!("{method} {url}: {phase}: {e}")
    }
}

/// The real HTTP transport: drives the running engine (the `ignis-server`'s
/// OpenAI-compatible API) and measures the per-request timing.
///
/// `POST /v1/chat/completions` (streaming SSE for per-token timing when the
/// trace line is streaming, a single JSON body otherwise) + `GET /v1/models`
/// (a readiness probe). One `reqwest::blocking::Client` is shared across the
/// driver's worker threads (`Client` is cheap to share), so
/// `HttpEndpoint` is `Send + Sync` and fits the `Endpoint` seam as-is.
///
/// Every client this type builds sets its deadline **explicitly** (see
/// [`DEFAULT_REQUEST_TIMEOUT`]): the transport of a measurement harness must
/// never be the thing that decides how long a request is allowed to take.
#[derive(Debug, Clone)]
pub struct HttpEndpoint {
    /// The engine's base URL (e.g. `http://127.0.0.1:8080`).
    pub base_url: String,
    /// The shared HTTP client (cheap to share across the driver's worker
    /// threads).
    client: Client,
    /// The deadline `client` was built with (`None`: no deadline) — kept so
    /// a caller can report what the run is actually configured for.
    request_timeout: Option<Duration>,
    /// Send a trace line whose prompt is a recorded chat request body as
    /// that body (`--as-recorded`), not as one flat user message.
    as_recorded: bool,
    /// Resend a `503` `engine_full` (`--retry-engine-full`); `None`: record
    /// the refusal as a failure.
    retry: Option<RetryPolicy>,
    /// The EOS control this endpoint's engine is sent, once learned
    /// ([`HttpEndpoint::eos_control`]).
    eos_control: std::sync::OnceLock<EosControl>,
}

impl HttpEndpoint {
    /// An endpoint with the default deadline, or [`REQUEST_TIMEOUT_ENV`]'s
    /// when the environment sets one.
    pub fn new(base_url: impl Into<String>) -> Self {
        let raw = std::env::var(REQUEST_TIMEOUT_ENV).ok();
        Self::with_timeout(base_url, timeout_from_env(raw.as_deref()))
    }

    /// An endpoint with an explicit per-request deadline (`None`: none at
    /// all). The timeout covers the whole request *including reading the
    /// response body*, which for a streamed measurement request is its
    /// entire life — prefill, decode and all.
    pub fn with_timeout(base_url: impl Into<String>, request_timeout: Option<Duration>) -> Self {
        let client = Client::builder()
            .timeout(request_timeout)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .expect("the bench HTTP client builds");
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client,
            request_timeout,
            as_recorded: false,
            retry: None,
            eos_control: std::sync::OnceLock::new(),
        }
    }

    /// Replay a line whose prompt parses as a chat request body (an object
    /// with `messages`) as that body — see [`as_recorded_body`]. Any other
    /// line is sent as before. Off by default: a baseline recorded with the
    /// flat message is only comparable to a replay that sends it.
    pub fn as_recorded(mut self, on: bool) -> Self {
        self.as_recorded = on;
        self
    }

    /// Wait out and resend a `503` `engine_full` per `policy` instead of
    /// recording it as a failure.
    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = Some(policy);
        self
    }

    /// The per-request deadline this endpoint drives the engine with
    /// (`None`: none at all).
    pub fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
    }

    /// `GET /v1/models` — a readiness probe: the servable model ids in the
    /// server's order, the loaded one first (spec model-switch/03).
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        let value = self.get_json("/v1/models")?;
        value
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string))
                    .collect()
            })
            .ok_or_else(|| format!("GET {}/v1/models: no models in the list", self.base_url))
    }

    /// `GET <base><path>`'s body, on this endpoint's deadline: the server's
    /// other routes as a preset reads them (`small_card`), and the metrics
    /// listener through an endpoint of its own base. A status other than 2xx
    /// is an `Err` naming it.
    pub fn get_text(&self, path: &str) -> Result<String, String> {
        let url = format!("{}{path}", self.base_url);
        self.client
            .get(&url)
            .send()
            .map_err(|e| transport_error("GET", &url, "send", &e))?
            .error_for_status()
            .map_err(|e| format!("GET {url} -> {e}"))?
            .text()
            .map_err(|e| transport_error("GET", &url, "read", &e))
    }

    /// The EOS control a measurement lane sent to this endpoint carries
    /// (GitHub #426), learned from the engine once and kept: the `owned_by`
    /// of the first model `GET /v1/models` lists, read by
    /// [`EosControl::for_owner`]. A request that suppresses nothing never
    /// asks.
    pub fn eos_control(&self) -> Result<EosControl, String> {
        if let Some(control) = self.eos_control.get() {
            return Ok(*control);
        }
        let models = self.get_json("/v1/models")?;
        let owner = models.pointer("/data/0/owned_by").and_then(|owner| owner.as_str());
        Ok(*self.eos_control.get_or_init(|| EosControl::for_owner(owner)))
    }

    /// [`HttpEndpoint::get_text`], parsed as JSON.
    pub fn get_json(&self, path: &str) -> Result<serde_json::Value, String> {
        let text = self.get_text(path)?;
        serde_json::from_str(&text).map_err(|e| format!("GET {}{path}: parse: {e}", self.base_url))
    }

    /// `POST <base><path>` with the JSON `body`, the answer parsed as JSON,
    /// on this endpoint's deadline. A status other than 2xx is an `Err`
    /// naming it and the server's `error.message`.
    pub fn post_json(&self, path: &str, body: &serde_json::Value) -> Result<serde_json::Value, String> {
        let url = format!("{}{path}", self.base_url);
        let response = self.client.post(&url).json(body).send().map_err(|e| transport_error("POST", &url, "send", &e))?;
        let status = response.status();
        let text = response.text().map_err(|e| transport_error("POST", &url, "read", &e))?;
        let value: Option<serde_json::Value> = serde_json::from_str(&text).ok();
        if !status.is_success() {
            let message = value.as_ref().and_then(|v| v.pointer("/error/message")).and_then(|m| m.as_str()).unwrap_or(&text);
            return Err(format!("POST {url} -> {status}: {message}"));
        }
        value.ok_or_else(|| format!("POST {url}: the answer is not JSON"))
    }
}

/// The `/v1/chat/completions` JSON body for `req` (GitHub #68: `enable_thinking`
/// rides along only when the request set it — most replay traffic leaves it
/// unset, letting the server's configured default apply).
fn request_body(req: &Request) -> serde_json::Value {
    let content = if req.images.is_empty() {
        serde_json::json!(req.prompt)
    } else {
        let mut parts = vec![serde_json::json!({ "type": "text", "text": req.prompt })];
        parts.extend(
            req.images
                .iter()
                .map(|url| serde_json::json!({ "type": "image_url", "image_url": { "url": url } })),
        );
        serde_json::Value::Array(parts)
    };
    let mut body = serde_json::json!({
        "messages": [{ "role": "user", "content": content }],
        "max_tokens": req.max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": req.stream,
        // GitHub #120: the trace's own main/sub class survives replay into
        // the engine's `class` ignis extension, rather than being dropped
        // at the client — the engine's request log then attributes a
        // per-class gate cell without a second run.
        "class": req.class.as_ignis_extension_str(),
    });
    if let Some(enable_thinking) = req.enable_thinking {
        body["enable_thinking"] = serde_json::json!(enable_thinking);
    }
    // The trailing usage chunk (OpenAI's `stream_options.include_usage`):
    // meaningless on a non-streaming response, which always carries `usage`.
    if req.include_usage && req.stream {
        body["stream_options"] = serde_json::json!({ "include_usage": true });
    }
    body
}

/// The body of a trace line whose `prompt` is a recorded chat request body
/// (`--as-recorded`): the JSON a real client sent, so the engine sees the
/// conversation, the tools and the other request fields it saw then.
/// `None` when the prompt is not an object with a `messages` array.
///
/// Everything the recorded body carries is kept except what the harness
/// owns, which it sets as [`request_body`] does: `max_tokens` (the trace
/// line's; the recorded `max_completion_tokens` is dropped so the two cannot
/// disagree), `stream` and `stream_options`, `temperature: 0` and `seed: 0`
/// (the greedy contract of the gate), the `class` extension and
/// `enable_thinking` when the request sets it. The recorded `model` is
/// dropped: which model answers is the server's configuration, not the
/// trace's.
fn as_recorded_body(req: &Request) -> Option<serde_json::Value> {
    let mut body = recorded_chat_body(&req.prompt)?;
    let object = body.as_object_mut()?;
    object.remove("model");
    object.remove("max_completion_tokens");
    object.remove("stream_options");
    object.insert("max_tokens".into(), serde_json::json!(req.max_tokens));
    object.insert("temperature".into(), serde_json::json!(0));
    object.insert("seed".into(), serde_json::json!(0));
    object.insert("stream".into(), serde_json::json!(req.stream));
    object.insert("class".into(), serde_json::json!(req.class.as_ignis_extension_str()));
    if let Some(enable_thinking) = req.enable_thinking {
        object.insert("enable_thinking".into(), serde_json::json!(enable_thinking));
    }
    if req.include_usage && req.stream {
        object.insert("stream_options".into(), serde_json::json!({ "include_usage": true }));
    }
    Some(body)
}

/// A trace line's `prompt` as a recorded chat request: a JSON object with a
/// `messages` array. Anything else is sent flat, even under `--as-recorded`.
fn recorded_chat_body(prompt: &str) -> Option<serde_json::Value> {
    let body: serde_json::Value = serde_json::from_str(prompt).ok()?;
    body.get("messages")?.is_array().then_some(body)
}

/// Whether `--as-recorded` sends this `prompt` as the recorded body (and
/// not as one flat user message).
pub fn is_recorded_chat_body(prompt: &str) -> bool {
    recorded_chat_body(prompt).is_some()
}

/// The end-of-sequence control a measurement lane's body carries (GitHub
/// #426): each engine is sent the one control it is asked by, never both,
/// because ignis refuses the reference's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EosControl {
    /// `ignore_eos: true`, ignis's extension for bounded measurement streams.
    /// ignis refuses a non-empty `logit_bias` with a 400 (#284, #288), so the
    /// body carries no `logit_bias` at all.
    IgnoreEos,
    /// OpenAI's `logit_bias`, -100 on every EOS id: the reference's control,
    /// and the standard field for any engine that is not ignis. The reference
    /// accepts it without applying it (finding 2026-09-13, "ITL lane
    /// terminator race"), so its lanes can still end at their own EOS.
    LogitBias,
}

impl EosControl {
    /// The control an engine is sent, from the owner `GET /v1/models` names
    /// for its models (`owned_by`): `"ignis"` on ignis, `"ninfer"` on the
    /// reference. Anything but ignis, a missing owner included, gets the
    /// OpenAI field.
    pub fn for_owner(owned_by: Option<&str>) -> Self {
        match owned_by {
            Some("ignis") => Self::IgnoreEos,
            _ => Self::LogitBias,
        }
    }

    /// The request field this control sets.
    pub fn field(self) -> &'static str {
        match self {
            Self::IgnoreEos => "ignore_eos",
            Self::LogitBias => "logit_bias",
        }
    }
}

#[cfg(test)]
fn request_body_suppressing_eos(req: &Request, eos: &[u32], control: EosControl) -> serde_json::Value {
    suppress_eos_in(request_body(req), eos, control)
}

/// `body` with the end-of-sequence tokens in `eos` suppressed through
/// `control` alone. An empty `eos` adds nothing.
fn suppress_eos_in(mut body: serde_json::Value, eos: &[u32], control: EosControl) -> serde_json::Value {
    if eos.is_empty() {
        return body;
    }
    match control {
        EosControl::IgnoreEos => body["ignore_eos"] = serde_json::json!(true),
        EosControl::LogitBias => {
            body["logit_bias"] = serde_json::Value::Object(
                eos.iter()
                    .map(|id| (id.to_string(), serde_json::json!(-100.0)))
                    .collect(),
            );
        }
    }
    body
}

/// The `usage` figures the TTFT instrument reads back: the prompt tokens
/// the engine counted, and how many of them it served from a prefix cache
/// (OpenAI's `prompt_tokens_details.cached_tokens`, which the reference
/// reports and ignis simply omits — it has no prefix cache before G4).
struct Usage {
    prompt_tokens: Option<u32>,
    cached_prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

impl Usage {
    /// Read the `usage` object out of a `chat.completion` body or a
    /// `chat.completion.chunk`'s trailing usage chunk — both carry `usage`
    /// at the same JSON path.
    fn from_response(value: &serde_json::Value) -> Self {
        let as_u32 = |v: Option<&serde_json::Value>| v.and_then(|v| v.as_u64()).map(|v| v as u32);
        Self {
            prompt_tokens: as_u32(value.pointer("/usage/prompt_tokens")),
            cached_prompt_tokens: as_u32(value.pointer("/usage/prompt_tokens_details/cached_tokens")),
            completion_tokens: as_u32(value.pointer("/usage/completion_tokens")),
        }
    }
}

impl Endpoint for HttpEndpoint {
    fn complete(&self, req: &Request) -> Result<Outcome, String> {
        self.complete_observed(req, &mut |_| {})
    }

    fn complete_queued(&self, req: &Request) -> (Result<Outcome, String>, Queueing) {
        self.complete_counting(req, &mut |_| true, &[])
    }

    fn complete_observed_while(
        &self,
        req: &Request,
        observer: &mut dyn FnMut(f64) -> bool,
        suppress_eos: &[u32],
    ) -> Result<Outcome, String> {
        self.complete_counting(req, observer, suppress_eos).0
    }
}

impl HttpEndpoint {
    /// The request body this endpoint sends for `req`, suppressing
    /// `suppress_eos` with the control its engine is sent
    /// ([`HttpEndpoint::eos_control`]).
    fn wire_body(&self, req: &Request, suppress_eos: &[u32]) -> Result<serde_json::Value, String> {
        // The trace line's prompt becomes a single user message (the trace
        // format is prompt-based; the shared "system + tools" prefix is
        // carried inside the prompt text) — unless `--as-recorded` and the
        // prompt is a recorded chat body. `temperature: 0` + `seed: 0` pin
        // the greedy + fixed-seed contract of the v1 gate (ADR 0007 — the
        // server's defaults, sent explicitly).
        let base = self
            .as_recorded
            .then(|| as_recorded_body(req))
            .flatten()
            .unwrap_or_else(|| request_body(req));
        if suppress_eos.is_empty() {
            return Ok(base);
        }
        Ok(suppress_eos_in(base, suppress_eos, self.eos_control()?))
    }

    /// One request through to its outcome, resending an `engine_full`
    /// refusal when [`HttpEndpoint::with_retry`] asked for it. The
    /// [`Outcome`]'s timings start at the attempt that was admitted.
    fn complete_counting(
        &self,
        req: &Request,
        observer: &mut dyn FnMut(f64) -> bool,
        suppress_eos: &[u32],
    ) -> (Result<Outcome, String>, Queueing) {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut queueing = Queueing::default();
        let body = match self.wire_body(req, suppress_eos) {
            Ok(body) => body,
            Err(e) => return (Err(e), queueing),
        };
        loop {
            let start = Instant::now();
            let resp = match self.client.post(&url).json(&body).send() {
                Ok(resp) => resp,
                Err(e) => {
                    let msg = if e.is_timeout() {
                        deadline_elapsed("POST", &url, "send")
                    } else {
                        format!("POST {url} failed: {e}")
                    };
                    return (Err(msg), queueing);
                }
            };
            let status = resp.status();
            if status.is_success() {
                let outcome = if req.stream {
                    self.read_sse(resp, start, observer)
                } else {
                    self.read_json(resp, start)
                };
                return (outcome, queueing);
            }
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let detail = resp.text().unwrap_or_default();
            let wait = self.retry.and_then(|policy| {
                (status == reqwest::StatusCode::SERVICE_UNAVAILABLE
                    && is_engine_full(&detail)
                    && queueing.retries < policy.max_retries)
                    .then(|| policy.next_wait(retry_after.as_deref(), queueing.wait_ms))
                    .flatten()
            });
            match wait {
                Some(wait) => {
                    let slept = Instant::now();
                    std::thread::sleep(wait);
                    queueing.retries += 1;
                    queueing.wait_ms += ms_since(slept);
                }
                None => return (Err(format!("POST {url} -> {status}: {detail}")), queueing),
            }
        }
    }
}

impl HttpEndpoint {
    /// The non-streaming half: a single JSON body (the server's
    /// `chat.completion` shape). ttft == total (no per-token timing — the
    /// metrics model reports tok_s = 0 for a non-streaming request).
    fn read_json(&self, resp: Response, start: Instant) -> Result<Outcome, String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let value: serde_json::Value = resp.json().map_err(|e| {
            if e.is_timeout() {
                deadline_elapsed("POST", &url, "parse")
            } else {
                format!("POST {url}: parse: {e}")
            }
        })?;
        let total_ms = ms_since(start);
        let usage = Usage::from_response(&value);
        let finish_reason = value
            .pointer("/choices/0/finish_reason")
            .and_then(|v| v.as_str())
            .map(|reason| FinishReason::Engine(reason.to_string()));
        Ok(Outcome {
            ttft_ms: total_ms,
            total_ms,
            prompt_tokens: usage.prompt_tokens,
            cached_prompt_tokens: usage.cached_prompt_tokens,
            // The engine's own `completion_tokens` already counts every token
            // it generated, thinking included — nothing to add here.
            n_tokens: value
                .pointer("/usage/completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            output: value
                .pointer("/choices/0/message/content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            reasoning_output: value
                .pointer("/choices/0/message/reasoning_content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            // A single JSON body reports one token figure and no per-channel
            // split (GitHub #137): unknown, not zero.
            reasoning_tokens: None,
            // No per-token timing on a non-streaming response.
            token_times_ms: Vec::new(),
            finish_reason,
        })
    }

    /// The streaming half: the SSE `chat.completion.chunk` framing — a
    /// chunk per generated token (a token's delta), a finish chunk (an empty
    /// delta + `finish_reason`), a terminal `[DONE]` marker.
    ///
    /// Delegates to [`read_sse_stream`], which is the same reader over any
    /// `BufRead` — a recorded stream in a test, the live response here.
    fn read_sse(
        &self,
        resp: Response,
        start: Instant,
        observer: &mut dyn FnMut(f64) -> bool,
    ) -> Result<Outcome, String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        read_sse_stream(std::io::BufReader::new(resp), &url, start, observer)
    }
}

/// Read one SSE `chat.completion.chunk` stream into an [`Outcome`].
///
/// A token arrives on one of two channels: `delta.content` (the answer) or
/// `delta.reasoning_content` (the thinking channel both engines stream by
/// default). **Both are generated tokens** and both are counted, timed and
/// observed here — reading only `content` made a thinking-heavy agentic turn
/// look like a request that produced nothing at all, which is how the G4
/// per-class cell and the canary came to report `n_tokens = 0` and
/// `ttft_ms = total_ms` against streams that were perfectly well formed
/// (GitHub #137). The two texts stay apart on the `Outcome` so a correctness
/// check still reads the answer alone.
///
/// A `delta.tool_calls` chunk is deliberately *not* counted: ignis emits one
/// complete call per chunk while the reference streams its arguments in
/// fragments, so counting them would compare two different framings rather
/// than two decode rates. Such a chunk carries no text on either channel, so
/// it is simply passed over.
///
/// `url` names the endpoint in the error messages; `start` is the instant the
/// request was sent, which every reported time is relative to.
pub fn read_sse_stream<R: BufRead>(
    reader: R,
    url: &str,
    start: Instant,
    observer: &mut dyn FnMut(f64) -> bool,
) -> Result<Outcome, String> {
    let mut n_tokens: u32 = 0;
    let mut reasoning_tokens: u32 = 0;
    let mut output = String::new();
    let mut reasoning_output = String::new();
    let mut first_token_ms: Option<f64> = None;
    let mut prompt_tokens: Option<u32> = None;
    let mut cached_prompt_tokens: Option<u32> = None;
    let mut token_times_ms: Vec<f64> = Vec::new();
    // GitHub #114. `engine_finish` is read from every chunk rather than
    // only the terminal one: the OpenAI shape puts it on a trailing
    // chunk with an empty delta, but an engine that hangs it off the
    // last content chunk is equally readable this way, last write
    // winning. `cancelled` records that *this* client ended the stream,
    // in which case the engine never got to name a reason at all.
    let mut engine_finish: Option<String> = None;
    let mut cancelled = false;
    // GitHub #159: a speculative engine may stream a verify round's whole
    // run as one chunk, so the chunk count undercounts it. The trailing
    // usage chunk's `completion_tokens` is the engine's own count.
    let mut usage_completion_tokens: Option<u32> = None;
    for line in reader.lines() {
        let line = line.map_err(|e| {
            if is_client_deadline(&e) {
                deadline_elapsed("POST", url, "read SSE")
            } else {
                format!("POST {url}: read SSE: {e}")
            }
        })?;
        // The SSE framing: `data: <payload>` lines (empty lines
        // separate the events — skipped).
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let chunk = serde_json::from_str::<serde_json::Value>(data)
            .map_err(|e| format!("POST {url}: bad SSE chunk {data}: {e}"))?;
        // The trailing usage chunk (`stream_options.include_usage`):
        // empty `choices`, populated `usage`. It arrives last, so
        // reading it here costs nothing and keeps one parse of the
        // stream.
        let usage = Usage::from_response(&chunk);
        if let Some(prompt) = usage.prompt_tokens {
            prompt_tokens = Some(prompt);
            cached_prompt_tokens = usage.cached_prompt_tokens;
        }
        if usage.completion_tokens.is_some() {
            usage_completion_tokens = usage.completion_tokens;
        }
        if let Some(reason) = chunk
            .pointer("/choices/0/finish_reason")
            .and_then(|v| v.as_str())
        {
            engine_finish = Some(reason.to_string());
        }
        // The token's text and the channel it came on. An empty delta is
        // the finish chunk, not a token; a `tool_calls`-only delta is a
        // chunk this reader passes over (see the doc comment).
        let text_at = |pointer| chunk.pointer(pointer).and_then(|v| v.as_str()).unwrap_or("");
        let content = text_at("/choices/0/delta/content");
        let reasoning = text_at("/choices/0/delta/reasoning_content");
        if content.is_empty() && reasoning.is_empty() {
            continue;
        }
        let now_ms = ms_since(start);
        if first_token_ms.is_none() {
            first_token_ms = Some(now_ms);
        }
        token_times_ms.push(now_ms);
        output.push_str(content);
        reasoning_output.push_str(reasoning);
        n_tokens += 1;
        // A chunk carrying both channels at once is one token on the
        // reasoning side of the ledger; the server never sends one
        // (GitHub #68) and an engine that did would still be counted once.
        if !reasoning.is_empty() {
            reasoning_tokens += 1;
        }
        if !observer(now_ms) {
            cancelled = true;
            break;
        }
    }
    let total_ms = ms_since(start);
    // No token chunk on either channel: nothing to measure (ttft = total,
    // tok_s = 0).
    let ttft_ms = first_token_ms.unwrap_or(total_ms);
    // A cancelled stream never reads the usage chunk; the chunk count is
    // then all there is.
    let n_tokens = match usage_completion_tokens {
        Some(engine_count) if !cancelled => engine_count,
        _ => n_tokens,
    };
    Ok(Outcome {
        ttft_ms,
        total_ms,
        n_tokens,
        output,
        reasoning_output,
        reasoning_tokens: Some(reasoning_tokens),
        prompt_tokens,
        cached_prompt_tokens,
        token_times_ms,
        // Cancellation wins: it is why the stream ended here, and any
        // reason the engine had was never sent.
        finish_reason: if cancelled {
            Some(FinishReason::Cancelled)
        } else {
            engine_finish.map(FinishReason::Engine)
        },
    })
}

/// Elapsed milliseconds since `start` (the timing unit of an `Outcome`).
fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Replay configuration.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Maximum number of in-flight requests (the "N" in the N-decode-lanes /
    /// ~10-subagent load).
    pub max_concurrency: usize,
    /// Scales the arrival offsets: 1.0 = real time, 0.0 = fire immediately
    /// (tests). See `replay`.
    pub time_scale: f64,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 8,
            time_scale: 1.0,
        }
    }
}

fn scaled(ms: u64, time_scale: f64) -> Duration {
    Duration::from_secs_f64((ms as f64) * time_scale / 1000.0)
}

/// Re-sends the whole trace against `ep` and returns the per-request metrics.
///
/// Requests are fed to a bounded pool of `max_concurrency` worker threads,
/// honoring each request's arrival offset (scaled by `cfg.time_scale`). This
/// models the "1 main + N subagents" concurrent load. Returns one
/// `RequestMetrics` per request (a failed request is recorded with
/// `ok == false`, never a panic).
pub fn replay(ep: Arc<dyn Endpoint>, trace: &Trace, cfg: &ReplayConfig) -> Vec<RequestMetrics> {
    let jobs: Vec<(Request, u64)> = trace
        .lines_sorted_by_arrival()
        .iter()
        .map(|l| (l.request(), l.t_arrive_ms))
        .collect();
    if jobs.is_empty() {
        return Vec::new();
    }

    let start = Instant::now();
    let (tx_job, rx_job) = mpsc::channel::<(Request, u64)>();
    let (tx_res, rx_res) = mpsc::channel::<RequestMetrics>();
    let rx = Arc::new(Mutex::new(rx_job));

    let n_workers = cfg.max_concurrency.clamp(1, jobs.len());
    let mut handles = Vec::with_capacity(n_workers);
    for _ in 0..n_workers {
        let rx = Arc::clone(&rx);
        let tx_res = tx_res.clone();
        let ep = Arc::clone(&ep);
        let cfg = cfg.clone();
        handles.push(std::thread::spawn(move || {
            loop {
                let job = match rx.lock().unwrap().recv() {
                    Ok(job) => job,
                    Err(_) => break, // channel closed and drained.
                };
                let (req, t_arrive_ms) = job;
                // Wait until the (scaled) arrival offset.
                let target = start + scaled(t_arrive_ms, cfg.time_scale);
                let now = Instant::now();
                if target > now {
                    std::thread::sleep(target - now);
                }
                let (result, queueing) = ep.complete_queued(&req);
                let metrics = match result {
                    Ok(o) => RequestMetrics {
                        retries: queueing.retries,
                        retry_wait_ms: queueing.wait_ms,
                        id: req.id,
                        class: req.class,
                        ttft_ms: o.ttft_ms,
                        n_tokens: o.n_tokens,
                        total_ms: o.total_ms,
                        // GitHub #147: a response that delivered no token on
                        // either channel did not complete a measurement, even
                        // on a 200. Its timing is kept for the record.
                        ok: o.n_tokens > 0,
                    },
                    Err(_e) => RequestMetrics {
                        retries: queueing.retries,
                        retry_wait_ms: queueing.wait_ms,
                        id: req.id,
                        class: req.class,
                        ttft_ms: 0.0,
                        n_tokens: 0,
                        total_ms: 0.0,
                        ok: false,
                    },
                };
                let _ = tx_res.send(metrics);
            }
        }));
    }

    let n_jobs = jobs.len();
    // Feed all jobs (in arrival order), then close the channel so the workers
    // drain and finish.
    for (req, t) in jobs {
        let _ = tx_job.send((req, t));
    }
    drop(tx_job);
    for handle in handles {
        let _ = handle.join();
    }

    // Collect exactly one result per job. Each job sends exactly one result
    // before its worker finishes (we joined all workers above), so `recv()`
    // `n_jobs` times is safe. We must NOT use `rx_res.iter()` here: the
    // original `tx_res` sender is still alive in this scope, so the channel
    // never closes and `iter()` would block forever.
    let mut results: Vec<RequestMetrics> = (0..n_jobs)
        .map(|_| rx_res.recv().expect("result channel closed early"))
        .collect();
    results.sort_by(|a, b| a.id.cmp(&b.id));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_omits_enable_thinking_when_unset() {
        let req = Request {
            id: "r".into(),
            class: RequestClass::Sub,
            prompt: "p".into(),
            max_tokens: 4,
            stream: false,
            include_usage: false,
            enable_thinking: None,
            images: Vec::new(),
        };
        assert!(request_body(&req).get("enable_thinking").is_none());
    }

    #[test]
    fn request_body_sends_enable_thinking_when_set() {
        let req = Request {
            id: "r".into(),
            class: RequestClass::Sub,
            prompt: "p".into(),
            max_tokens: 4,
            stream: false,
            include_usage: false,
            enable_thinking: Some(false),
            images: Vec::new(),
        };
        assert_eq!(request_body(&req)["enable_thinking"], false);
    }

    #[test]
    fn request_body_carries_the_ignis_class_extension_for_each_trace_class() {
        let mut req = Request {
            id: "main".into(),
            class: RequestClass::Main,
            prompt: "p".into(),
            max_tokens: 4,
            stream: false,
            include_usage: false,
            enable_thinking: None,
            images: Vec::new(),
        };
        assert_eq!(request_body(&req)["class"], "interactive");
        req.class = RequestClass::Sub;
        assert_eq!(request_body(&req)["class"], "agent");
    }

    fn measurement_lane() -> Request {
        Request {
            id: "itl-decode-0".into(),
            class: RequestClass::Main,
            prompt: "p".into(),
            max_tokens: 3_072,
            stream: true,
            include_usage: true,
            enable_thinking: Some(false),
            images: Vec::new(),
        }
    }

    /// GitHub #426: ignis refuses a non-empty `logit_bias` (#284, #288), so
    /// a lane that also carried it never decoded a token there.
    #[test]
    fn a_measurement_lane_for_ignis_carries_ignore_eos_and_no_logit_bias() {
        let body = request_body_suppressing_eos(&measurement_lane(), &[151_645, 151_643], EosControl::IgnoreEos);
        assert_eq!(body["ignore_eos"], true);
        assert!(body.get("logit_bias").is_none(), "ignis refuses a non-empty logit_bias: {body}");
    }

    #[test]
    fn a_measurement_lane_for_the_reference_suppresses_every_eos_with_logit_bias() {
        let body = request_body_suppressing_eos(&measurement_lane(), &[151_645, 151_643], EosControl::LogitBias);
        assert_eq!(body["logit_bias"], serde_json::json!({ "151645": -100.0, "151643": -100.0 }));
        assert!(body.get("ignore_eos").is_none(), "only the control the reference is sent: {body}");
    }

    #[test]
    fn only_an_engine_whose_models_ignis_owns_is_sent_ignore_eos() {
        assert_eq!(EosControl::for_owner(Some("ignis")), EosControl::IgnoreEos);
        assert_eq!(EosControl::for_owner(Some("ninfer")), EosControl::LogitBias, "the reference");
        assert_eq!(EosControl::for_owner(None), EosControl::LogitBias, "an engine that names no owner");
    }

    #[test]
    fn a_request_with_images_sends_its_prompt_then_each_image_as_content_parts() {
        let mut req = Request {
            id: "r".into(),
            class: RequestClass::Main,
            prompt: "a000 What does this screenshot show?".into(),
            max_tokens: 4,
            stream: true,
            include_usage: true,
            enable_thinking: Some(false),
            images: vec!["data:image/png;base64,AAAA".into(), "http://127.0.0.1/b.png".into()],
        };
        assert_eq!(
            request_body(&req)["messages"],
            serde_json::json!([{ "role": "user", "content": [
                { "type": "text", "text": "a000 What does this screenshot show?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
                { "type": "image_url", "image_url": { "url": "http://127.0.0.1/b.png" } },
            ]}])
        );
        req.images.clear();
        assert_eq!(
            request_body(&req)["messages"][0]["content"],
            "a000 What does this screenshot show?",
            "a text-only request keeps the plain-string content"
        );
    }

    #[test]
    fn request_body_asks_for_the_usage_chunk_only_when_streaming() {
        let base = Request {
            id: "r".into(),
            class: RequestClass::Sub,
            prompt: "p".into(),
            max_tokens: 4,
            stream: true,
            include_usage: true,
            enable_thinking: None,
            images: Vec::new(),
        };
        assert_eq!(
            request_body(&base)["stream_options"]["include_usage"],
            true,
            "a streaming request must ask for the trailing usage chunk"
        );
        // A non-streaming response always carries `usage`; asking for it
        // through `stream_options` would be a field the engine ignores.
        let non_streaming = Request { stream: false, ..base.clone() };
        assert!(request_body(&non_streaming).get("stream_options").is_none());
        // Replay traffic never asks for it.
        let plain = Request { include_usage: false, ..base };
        assert!(request_body(&plain).get("stream_options").is_none());
    }

    fn recorded_request(prompt: &str) -> Request {
        Request {
            id: "r".into(),
            class: RequestClass::Sub,
            prompt: prompt.into(),
            max_tokens: 7,
            stream: false,
            include_usage: true,
            enable_thinking: None,
            images: Vec::new(),
        }
    }

    #[test]
    fn as_recorded_body_keeps_the_recorded_fields_and_overrides_the_harness_owned_ones() {
        let prompt = serde_json::json!({
            "model": "m", "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{ "type": "function" }], "parallel_tool_calls": false,
            "max_completion_tokens": 5000, "temperature": 1.0, "seed": 9, "stream": true,
            "stream_options": { "include_usage": false, "other": 1 },
        })
        .to_string();
        let body = as_recorded_body(&recorded_request(&prompt)).expect("a chat body");
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body["tools"], serde_json::json!([{ "type": "function" }]));
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["max_tokens"], 7);
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("model").is_none());
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["seed"], 0);
        assert_eq!(body["stream"], false, "the trace line decides streaming");
        assert!(body.get("stream_options").is_none(), "meaningless on a non-streaming request");
        assert_eq!(body["class"], "agent");
        let streaming = Request { stream: true, enable_thinking: Some(false), ..recorded_request(&prompt) };
        let body = as_recorded_body(&streaming).unwrap();
        assert_eq!(body["stream_options"], serde_json::json!({ "include_usage": true }));
        assert_eq!(body["enable_thinking"], false);
    }

    #[test]
    fn as_recorded_body_is_none_for_a_prompt_that_is_not_a_chat_body() {
        for prompt in ["plain text", "[1,2]", "{\"messages\": \"x\"}", "{\"input\": []}", "", "{\"messages\": ["] {
            assert!(as_recorded_body(&recorded_request(prompt)).is_none(), "{prompt}");
        }
    }

    #[test]
    fn the_wait_is_retry_after_bounded_and_one_second_without_it() {
        let cap = Duration::from_secs(30);
        assert_eq!(retry_wait(Some("7"), cap), Duration::from_secs(7));
        assert_eq!(retry_wait(Some(" 60 "), cap), cap, "bounded");
        assert_eq!(retry_wait(None, cap), Duration::from_secs(1));
        assert_eq!(retry_wait(Some("Wed, 21 Oct"), cap), Duration::from_secs(1));
        assert_eq!(retry_wait(Some("0"), cap), Duration::ZERO);
    }

    #[test]
    fn the_total_wait_budget_cuts_the_last_wait_and_then_ends_the_retries() {
        let policy = RetryPolicy {
            max_retries: 100,
            max_wait: Duration::from_secs(30),
            total_wait: Duration::from_secs(45),
        };
        assert_eq!(policy.next_wait(Some("20"), 0.0), Some(Duration::from_secs(20)));
        assert_eq!(policy.next_wait(Some("20"), 20_000.0), Some(Duration::from_secs(20)));
        assert_eq!(policy.next_wait(Some("20"), 40_000.0), Some(Duration::from_secs(5)), "cut to what is left");
        assert_eq!(policy.next_wait(Some("20"), 45_000.0), None, "spent");
        assert_eq!(policy.next_wait(None, 60_000.0), None);
    }

    #[test]
    fn only_the_engine_full_code_is_a_refusal_to_wait_out() {
        assert!(is_engine_full(r#"{"error":{"code":"engine_full","message":"m"}}"#));
        assert!(!is_engine_full(r#"{"error":{"code":"model_switching"}}"#));
        assert!(!is_engine_full("engine_full"));
    }

    #[test]
    fn the_driver_records_the_queueing_an_endpoint_reports() {
        struct Queued;
        impl Endpoint for Queued {
            fn complete(&self, _: &Request) -> Result<Outcome, String> {
                unreachable!("the driver asks for the queueing")
            }
            fn complete_queued(&self, req: &Request) -> (Result<Outcome, String>, Queueing) {
                let q = Queueing { retries: 2, wait_ms: 1500.0 };
                match req.id.as_str() {
                    "main" => (Ok(outcome(100.0, 1000.0, 10)), q),
                    _ => (Err("engine_full".into()), q),
                }
            }
        }
        let trace = Trace::from_jsonl(&trace_jsonl()).expect("valid trace");
        let results = replay(Arc::new(Queued), &trace, &ReplayConfig { max_concurrency: 2, time_scale: 0.0 });
        for m in &results {
            assert_eq!((m.retries, m.retry_wait_ms), (2, 1500.0), "{}", m.id);
        }
        assert!(results.iter().find(|m| m.id == "main").unwrap().ok);
        assert!(!results.iter().find(|m| m.id == "s1").unwrap().ok, "a failure keeps its queueing too");
    }

    #[test]
    fn the_transport_never_takes_reqwests_own_deadline() {
        // GitHub #138: `Client::new()`'s default is 30 s, covering the body
        // read — the G4 needle cell at 131,072 tokens died on it while the
        // engine was still prefilling. Whatever this default is, it must be
        // far enough above a real measurement request not to be one.
        assert!(
            DEFAULT_REQUEST_TIMEOUT >= Duration::from_secs(10 * 60),
            "{DEFAULT_REQUEST_TIMEOUT:?} is close enough to a real request to cut one"
        );
    }

    #[test]
    fn the_run_can_raise_or_remove_the_deadline() {
        assert_eq!(
            timeout_from_env(None),
            Some(DEFAULT_REQUEST_TIMEOUT),
            "an unset variable takes the default"
        );
        assert_eq!(timeout_from_env(Some("")), Some(DEFAULT_REQUEST_TIMEOUT));
        assert_eq!(timeout_from_env(Some("  ")), Some(DEFAULT_REQUEST_TIMEOUT));
        assert_eq!(
            timeout_from_env(Some(" 900 ")),
            Some(Duration::from_secs(900)),
            "a whole number of seconds is the deadline"
        );
        assert_eq!(
            timeout_from_env(Some("0")),
            None,
            "0 removes the deadline — the harness waits as long as the engine takes"
        );
        assert_eq!(
            timeout_from_env(Some("15m")),
            Some(DEFAULT_REQUEST_TIMEOUT),
            "an unparseable value falls back to the default rather than to no deadline"
        );
    }

    /// A usage chunk that arrives after the tokens, as a stream sends it.
    const STREAM_WITH_USAGE: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":9}}\n\n",
        "data: [DONE]\n\n",
    );

    #[test]
    fn a_stream_that_ran_to_its_end_reports_the_engines_own_token_count() {
        let out = read_sse_stream(STREAM_WITH_USAGE.as_bytes(), "u", Instant::now(), &mut |_| true).unwrap();
        assert_eq!(out.n_tokens, 9, "usage, not the 3 chunks");
        assert_eq!(out.token_times_ms.len(), 3);
    }

    #[test]
    fn a_cancelled_stream_keeps_the_chunk_count() {
        // The client stops after two chunks, so the usage chunk is never
        // read and the count is the chunks seen (GitHub #162).
        let mut seen = 0;
        let out = read_sse_stream(STREAM_WITH_USAGE.as_bytes(), "u", Instant::now(), &mut |_| {
            seen += 1;
            seen < 2
        })
        .unwrap();
        assert_eq!(out.n_tokens, 2);
        assert_eq!(out.token_times_ms.len(), 2);
        assert_eq!(out.finish_reason, Some(FinishReason::Cancelled));
    }

    #[test]
    fn computed_prefill_subtracts_the_cached_prompt_tokens() {
        let mut o = MockEndpoint::fallback_for(&Request {
            id: "r".into(),
            class: RequestClass::Sub,
            prompt: "one two three".into(),
            max_tokens: 1,
            stream: false,
            include_usage: false,
            enable_thinking: None,
            images: Vec::new(),
        });
        assert_eq!(o.computed_prefill_tokens(), Some(3));
        o.cached_prompt_tokens = Some(2);
        assert_eq!(o.computed_prefill_tokens(), Some(1), "a cache hit is not computed prefill");
        o.prompt_tokens = None;
        assert_eq!(
            o.computed_prefill_tokens(),
            None,
            "an engine that reports no usage cannot prove a cold prefix"
        );
    }

    fn trace_jsonl() -> String {
        [
            r#"{"id":"main","class":"main","t_arrive_ms":0,"prompt":"P","max_tokens":512,"stream":true}"#,
            r#"{"id":"s1","class":"sub","t_arrive_ms":50,"prompt":"Q","max_tokens":64,"stream":true}"#,
            r#"{"id":"s2","class":"sub","t_arrive_ms":50,"prompt":"Q","max_tokens":64,"stream":true}"#,
        ]
        .join("\n")
    }

    fn outcome(ttft: f64, total: f64, n: u32) -> Outcome {
        Outcome {
            ttft_ms: ttft,
            total_ms: total,
            n_tokens: n,
            output: "x".repeat(n as usize),
            reasoning_output: String::new(),
            reasoning_tokens: Some(0),
            prompt_tokens: None,
            cached_prompt_tokens: None,
            token_times_ms: Vec::new(),
            finish_reason: None,
        }
    }

    #[test]
    fn replay_collects_metrics_for_every_request() {
        let trace = Trace::from_jsonl(&trace_jsonl()).expect("valid trace");
        // Canned: one outcome per request (order doesn't matter — matched by id
        // after the driver sorts results).
        let ep = Arc::new(MockEndpoint::new(vec![
            outcome(100.0, 2000.0, 10),
            outcome(120.0, 1500.0, 20),
            outcome(90.0, 900.0, 5),
        ]));
        let cfg = ReplayConfig {
            max_concurrency: 2,
            time_scale: 0.0, // no real waiting (fast test).
        };
        let results = replay(ep.clone(), &trace, &cfg);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|m| m.ok));
        // Every request was actually sent to the endpoint.
        assert_eq!(ep.received.lock().unwrap().len(), 3);
    }

    #[test]
    fn results_are_sorted_by_id_and_tok_s_is_computed() {
        // Keyed by id, not `MockEndpoint`'s canned queue: the queue is
        // consumed by whichever worker calls `complete` first, and with
        // `max_concurrency: 4` and `time_scale: 0.0` all three requests are
        // in flight at once, so which one gets the "main" outcome is a race.
        // It resolved the same way on Windows for long enough to look
        // deterministic and came out the other way the first time this suite
        // ran on Linux (tok_s 70.0 — main had been handed s1's outcome).
        // This test is about tok_s, so it pins the mapping instead.
        struct ById;
        impl Endpoint for ById {
            fn complete(&self, req: &Request) -> Result<Outcome, String> {
                // "main": 512 tokens, ttft 200 ms, total 6200 ms -> 511
                // decode tokens / 6.0 s ~= 85.2 tok/s.
                Ok(match req.id.as_str() {
                    "main" => outcome(200.0, 6200.0, 512),
                    _ => outcome(100.0, 1000.0, 64),
                })
            }
        }
        let trace = Trace::from_jsonl(&trace_jsonl()).expect("valid trace");
        let ep = Arc::new(ById);
        let cfg = ReplayConfig {
            max_concurrency: 4,
            time_scale: 0.0,
        };
        let results = replay(ep, &trace, &cfg);
        let ids: Vec<_> = results.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["main", "s1", "s2"]);
        let main = results.iter().find(|m| m.id == "main").expect("main");
        assert!((main.tok_s() - (511.0 / 6.0)).abs() < 1e-3, "tok_s {}", main.tok_s());
    }

    #[test]
    fn a_failed_request_is_recorded_not_panicked() {
        let trace = Trace::from_jsonl(&trace_jsonl()).expect("valid trace");
        // An endpoint that always fails.
        struct Failing;
        impl Endpoint for Failing {
            fn complete(&self, _req: &Request) -> Result<Outcome, String> {
                Err("endpoint down".into())
            }
        }
        let ep = Arc::new(Failing);
        let cfg = ReplayConfig {
            max_concurrency: 2,
            time_scale: 0.0,
        };
        let results = replay(ep, &trace, &cfg);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|m| !m.ok && m.n_tokens == 0));
    }

    #[test]
    fn a_response_that_generated_no_tokens_is_not_recorded_as_ok() {
        // GitHub #147: a 200 response whose stream carried no token on either
        // channel (`ttft_ms == total_ms`, `n_tokens == 0`) used to read back
        // as `ok: true`, and the G4 `main` cell ranked it as 0.000 tok/s.
        let trace = Trace::from_jsonl(&trace_jsonl()).expect("valid trace");
        let ep = Arc::new(MockEndpoint::new(vec![
            outcome(451_653.7, 451_653.7, 0),
            outcome(100.0, 1000.0, 64),
            outcome(100.0, 1000.0, 64),
        ]));
        let cfg = ReplayConfig {
            max_concurrency: 1,
            time_scale: 0.0,
        };
        let results = replay(ep, &trace, &cfg);
        let empty = results.iter().find(|m| m.n_tokens == 0).expect("the empty response");
        assert!(!empty.ok, "{empty:?}");
        assert_eq!(empty.total_ms, 451_653.7, "the wall time it cost is kept");
        assert_eq!(results.iter().filter(|m| m.ok).count(), 2);
    }

    #[test]
    fn a_single_token_or_no_decode_gives_zero_toks() {
        let req = Request {
            id: "r".into(),
            class: RequestClass::Sub,
            prompt: "p".into(),
            max_tokens: 1,
            stream: false,
            include_usage: false,
            enable_thinking: None,
            images: Vec::new(),
        };
        let o = MockEndpoint::fallback_for(&req);
        let m = RequestMetrics {
            id: "r".into(),
            class: RequestClass::Sub,
            ttft_ms: o.ttft_ms,
            n_tokens: o.n_tokens,
            total_ms: o.total_ms,
            ok: true,
            retries: 0,
            retry_wait_ms: 0.0,
        };
        assert_eq!(m.tok_s(), 0.0);
    }

    // ── the SSE reader, against recorded streams (GitHub #137) ───────────

    /// One `data:` line of a `chat.completion.chunk` carrying `delta`.
    fn sse_chunk(delta: serde_json::Value) -> String {
        let chunk = serde_json::json!({
            "id": "cmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
        });
        format!("data: {chunk}\n\n")
    }

    /// Read a recorded SSE stream the way the live transport reads the wire.
    fn read_recorded(stream: &str) -> Outcome {
        read_sse_stream(
            std::io::Cursor::new(stream.to_string()),
            "http://recorded/v1/chat/completions",
            Instant::now(),
            &mut |_| true,
        )
        .expect("the recorded stream parses")
    }

    /// The shape the G4 dogfood actually recorded: a long thinking trace, a
    /// tool call, a clean finish — and not one `delta.content` chunk.
    fn thinking_then_tool_call_stream(n_reasoning: u32) -> String {
        let mut sse = String::new();
        for i in 0..n_reasoning {
            sse.push_str(&sse_chunk(
                serde_json::json!({ "reasoning_content": format!("think-{i} ") }),
            ));
        }
        sse.push_str(&sse_chunk(serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "id": "call_0",
                "type": "function",
                "function": { "name": "read_file", "arguments": "{\"path\":\"src/lib.rs\"}" },
            }],
        })));
        let finish = serde_json::json!({
            "id": "cmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "tool_calls" }],
        });
        sse.push_str(&format!("data: {finish}\n\ndata: [DONE]\n\n"));
        sse
    }

    #[test]
    fn a_tool_call_only_reply_with_usage_has_no_decode_window_and_stays_out_of_the_rate() {
        // `--as-recorded` sends `tools`: a reply may be one `tool_calls` delta
        // and nothing else. Its token count is the engine's (`usage`), but
        // no text chunk timed a first token, so ttft == total.
        let mut sse = sse_chunk(serde_json::json!({
            "tool_calls": [{ "index": 0, "id": "call_0", "type": "function",
                "function": { "name": "read_file", "arguments": "{}" } }],
        }));
        sse.push_str(&format!(
            "data: {}

data: {}

data: [DONE]

",
            serde_json::json!({ "choices": [{ "index": 0, "delta": {}, "finish_reason": "tool_calls" }] }),
            serde_json::json!({ "choices": [], "usage": { "prompt_tokens": 50, "completion_tokens": 40 } }),
        ));
        let out = read_recorded(&sse);
        assert_eq!(out.n_tokens, 40, "the engine's own count");
        assert_eq!(out.ttft_ms, out.total_ms, "no text chunk to time");
        let calls = RequestMetrics {
            id: "calls".into(),
            class: RequestClass::Sub,
            ttft_ms: out.ttft_ms,
            n_tokens: out.n_tokens,
            total_ms: out.total_ms,
            ok: true,
            retries: 0,
            retry_wait_ms: 0.0,
        };
        let text = RequestMetrics { id: "text".into(), ttft_ms: 100.0, n_tokens: 11, total_ms: 1100.0, ..calls.clone() };
        let stats = crate::metrics::class_stats(RequestClass::Sub, &[calls, text]);
        assert_eq!(stats.n_requests, 2);
        assert_eq!(stats.total_tokens, 51);
        assert!((stats.tok_s - 10.0).abs() < 1e-9, "10 decoded tokens in 1 s, not 49: {}", stats.tok_s);
    }

    #[test]
    fn a_turn_that_answers_entirely_in_the_thinking_channel_is_still_measured() {
        // GitHub #137: this exact stream (100% `reasoning_content` + a
        // `tool_calls` delta) used to read back as `n_tokens = 0` and
        // `ttft_ms = total_ms`, which turned a G4 per-class throughput cell
        // into a fiction.
        let out = read_recorded(&thinking_then_tool_call_stream(12));
        assert_eq!(out.n_tokens, 12, "every thinking token is a generated token");
        assert_eq!(out.reasoning_tokens, Some(12), "all of them on the reasoning channel");
        assert_eq!(
            out.token_times_ms.len(),
            12,
            "the decode cadence is sampled on the thinking channel too"
        );
        assert_eq!(
            out.ttft_ms,
            out.token_times_ms[0],
            "ttft is the first generated token, not the 'nothing to measure' fallback"
        );
        assert!(
            out.ttft_ms <= out.total_ms,
            "ttft {} ms cannot outlast the request ({} ms)",
            out.ttft_ms,
            out.total_ms
        );
        // The answer channel stays empty: scratch work is not an answer.
        assert!(out.output.is_empty(), "no content token was sent");
        assert!(out.reasoning_output.contains("think-11"), "the thinking text is kept");
        assert_eq!(
            out.finish_reason,
            Some(FinishReason::Engine("tool_calls".into())),
            "the engine's own reason survives the tool-call chunk"
        );
    }

    #[test]
    fn the_two_channels_are_counted_together_and_reported_apart() {
        let sse = format!(
            "{}{}{}data: [DONE]\n\n",
            sse_chunk(serde_json::json!({ "reasoning_content": "let me think " })),
            sse_chunk(serde_json::json!({ "reasoning_content": "about it " })),
            sse_chunk(serde_json::json!({ "content": "42" })),
        );
        let out = read_recorded(&sse);
        assert_eq!(out.n_tokens, 3, "throughput counts every token the engine decoded");
        assert_eq!(out.reasoning_tokens, Some(2), "the split stays readable");
        assert_eq!(out.output, "42", "the answer alone is what a correctness check reads");
        assert_eq!(out.reasoning_output, "let me think about it ");
    }

    /// #159 C: under DFlash2 the reference streams a verify round's run as
    /// one chunk. The trailing usage chunk's `completion_tokens` is the
    /// engine's own count and wins over the chunk count.
    #[test]
    fn the_usage_chunk_counts_tokens_a_multi_token_chunk_would_hide() {
        let usage = serde_json::json!({
            "id": "cmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [],
            "usage": { "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110 },
        });
        let sse = format!(
            "{}{}{}{}data: {usage}\n\ndata: [DONE]\n\n",
            sse_chunk(serde_json::json!({ "content": "one" })),
            sse_chunk(serde_json::json!({ "content": " two three four" })),
            sse_chunk(serde_json::json!({ "content": " five six" })),
            sse_chunk(serde_json::json!({ "content": " seven eight nine ten" })),
        );
        let out = read_recorded(&sse);
        assert_eq!(out.n_tokens, 10, "the engine's count, not the four chunks");
        assert_eq!(out.token_times_ms.len(), 4, "arrival times stay per chunk");
    }

    #[test]
    fn an_observer_can_stop_a_stream_on_a_thinking_token() {
        // The G3 decode lanes stop their own stream at the measurement
        // boundary. A lane that thinks must be stoppable the same way.
        let sse = thinking_then_tool_call_stream(50);
        let mut seen = 0u32;
        let out = read_sse_stream(
            std::io::Cursor::new(sse),
            "http://recorded/v1/chat/completions",
            Instant::now(),
            &mut |_| {
                seen += 1;
                seen < 5
            },
        )
        .expect("the recorded stream parses");
        assert_eq!(out.n_tokens, 5, "the reader stopped where the observer said");
        assert_eq!(out.finish_reason, Some(FinishReason::Cancelled));
    }
}
