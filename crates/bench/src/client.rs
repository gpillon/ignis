//! The replay client: drives the engine endpoint and collects per-request
//! metrics.
//!
//! The harness talks to an OpenAI-compatible engine over HTTP. The I/O is
//! isolated behind the `Endpoint` trait so the replay driver + metrics logic
//! is testable against a `MockEndpoint` with **no** running server. The real
//! `HttpEndpoint` (a `reqwest` blocking client) drives the running
//! `ignis-server` endpoint (`POST /v1/chat/completions` — streaming SSE for
//! per-token timing + non-streaming — and `GET /v1/models`) and measures the
//! per-request timing (see `.scratch/bench/issues/01-trace-replay.md`).
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
}

/// The raw outcome of a single request completion.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// Time to first token (ms).
    pub ttft_ms: f64,
    /// Wall-clock duration of the whole request (ms).
    pub total_ms: f64,
    /// Number of tokens generated.
    pub n_tokens: u32,
    /// The full generated text.
    pub output: String,
}

/// An endpoint the replay driver can send requests to. `Send + Sync` so a
/// single shared endpoint can be used by the driver's worker threads.
pub trait Endpoint: Send + Sync {
    /// Send one request and return its outcome. Returns a human-readable error
    /// string on failure (the driver records a failed request, not a panic).
    fn complete(&self, req: &Request) -> Result<Outcome, String>;
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
        Outcome {
            ttft_ms: ttft,
            total_ms: ttft + decode_ms,
            n_tokens: n,
            output: format!("[{}]", req.id),
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

/// The real HTTP transport: drives the running engine (the `ignis-server`'s
/// OpenAI-compatible API) and measures the per-request timing.
///
/// `POST /v1/chat/completions` (streaming SSE for per-token timing when the
/// trace line is streaming, a single JSON body otherwise) + `GET /v1/models`
/// (a readiness probe). One `reqwest::blocking::Client` is shared across the
/// driver's worker threads (`Client` is cheap to share), so
/// `HttpEndpoint` is `Send + Sync` and fits the `Endpoint` seam as-is.
#[derive(Debug, Clone)]
pub struct HttpEndpoint {
    /// The engine's base URL (e.g. `http://127.0.0.1:8080`).
    pub base_url: String,
    /// The shared HTTP client (cheap to share across the driver's worker
    /// threads).
    client: Client,
}

impl HttpEndpoint {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: Client::new(),
        }
    }

    /// `GET /v1/models` — a readiness probe: the loaded model id(s)
    /// (v1: a single model).
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        let url = format!("{}/v1/models", self.base_url);
        let value: serde_json::Value = self
            .client
            .get(&url)
            .send()
            .map_err(|e| format!("GET {url} failed: {e}"))?
            .error_for_status()
            .map_err(|e| format!("GET {url} -> {e}"))?
            .json()
            .map_err(|e| format!("GET {url}: parse: {e}"))?;
        value
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string))
                    .collect()
            })
            .ok_or_else(|| format!("GET {url}: no models in the list"))
    }
}

impl Endpoint for HttpEndpoint {
    fn complete(&self, req: &Request) -> Result<Outcome, String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        // The trace line's prompt becomes a single user message (the trace
        // format is prompt-based; the shared "system + tools" prefix is
        // carried inside the prompt text). `temperature: 0` + `seed: 0` pin
        // the greedy + fixed-seed contract of the v1 gate (ADR 0007 — the
        // server's defaults, sent explicitly).
        let body = serde_json::json!({
            "messages": [{ "role": "user", "content": req.prompt }],
            "max_tokens": req.max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": req.stream,
        });
        let start = Instant::now();
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| format!("POST {url} failed: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().unwrap_or_default();
            return Err(format!("POST {url} -> {status}: {detail}"));
        }
        if req.stream {
            self.read_sse(resp, start)
        } else {
            self.read_json(resp, start)
        }
    }
}

impl HttpEndpoint {
    /// The non-streaming half: a single JSON body (the server's
    /// `chat.completion` shape). ttft == total (no per-token timing — the
    /// metrics model reports tok_s = 0 for a non-streaming request).
    fn read_json(&self, resp: Response, start: Instant) -> Result<Outcome, String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let value: serde_json::Value =
            resp.json()
                .map_err(|e| format!("POST {url}: parse: {e}"))?;
        let total_ms = ms_since(start);
        Ok(Outcome {
            ttft_ms: total_ms,
            total_ms,
            n_tokens: value
                .pointer("/usage/completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            output: value
                .pointer("/choices/0/message/content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        })
    }

    /// The streaming half: the SSE `chat.completion.chunk` framing — a
    /// content chunk per token (a token's delta), a finish chunk (an empty
    /// delta + `finish_reason`), a terminal `[DONE]` marker. ttft is the
    /// first content chunk; the token count is the non-empty deltas.
    fn read_sse(&self, resp: Response, start: Instant) -> Result<Outcome, String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let reader = std::io::BufReader::new(resp);
        let mut n_tokens: u32 = 0;
        let mut output = String::new();
        let mut first_token_ms: Option<f64> = None;
        for line in reader.lines() {
            let line = line.map_err(|e| format!("POST {url}: read SSE: {e}"))?;
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
            // `choices[0].delta.content` (an empty delta = the finish
            // chunk, not a token).
            let delta = chunk
                .pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !delta.is_empty() {
                if first_token_ms.is_none() {
                    first_token_ms = Some(ms_since(start));
                }
                output.push_str(delta);
                n_tokens += 1;
            }
        }
        let total_ms = ms_since(start);
        // No content chunk: nothing to measure (ttft = total, tok_s = 0).
        let ttft_ms = first_token_ms.unwrap_or(total_ms);
        Ok(Outcome {
            ttft_ms,
            total_ms,
            n_tokens,
            output,
        })
    }
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
                let metrics = match ep.complete(&req) {
                    Ok(o) => RequestMetrics {
                        id: req.id,
                        class: req.class,
                        ttft_ms: o.ttft_ms,
                        n_tokens: o.n_tokens,
                        total_ms: o.total_ms,
                        ok: true,
                    },
                    Err(_e) => RequestMetrics {
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
        let trace = Trace::from_jsonl(&trace_jsonl()).expect("valid trace");
        // Give the "main" request a known ttft/total so its tok_s is
        // deterministic: 512 tokens, ttft 200 ms, total 6200 ms -> 511
        // decode tokens / 6.0 s ~= 85.2 tok/s.
        let ep = Arc::new(MockEndpoint::new(vec![
            outcome(200.0, 6200.0, 512), // main
            outcome(100.0, 1000.0, 64),  // s1
            outcome(100.0, 1000.0, 64),  // s2
        ]));
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
    fn a_single_token_or_no_decode_gives_zero_toks() {
        let req = Request {
            id: "r".into(),
            class: RequestClass::Sub,
            prompt: "p".into(),
            max_tokens: 1,
            stream: false,
        };
        let o = MockEndpoint::fallback_for(&req);
        let m = RequestMetrics {
            id: "r".into(),
            class: RequestClass::Sub,
            ttft_ms: o.ttft_ms,
            n_tokens: o.n_tokens,
            total_ms: o.total_ms,
            ok: true,
        };
        assert_eq!(m.tok_s(), 0.0);
    }
}