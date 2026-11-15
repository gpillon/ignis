//! The 27B served with its MTP head as the drafter (spec #618, GitHub #623),
//! through the server's own load path (`load::prepare_model`,
//! `load::load_model`) from the flags an operator passes, on the real NVFP4
//! artifact behind the production router. An MTP load serves what a DFlash2
//! load serves:
//!
//! 1. **Greedy text is the spec-off text, up to the near-tie rule.** The same
//!    prompts on an MTP load at its default window (7) and on the same load
//!    at a window of 0, which loads no speculation (the load `off` makes):
//!    token for token, or identical up to a first split where the window-0
//!    load's prefill route, after the agreed text, rates the MTP load's pick
//!    within ADR 0056's near tie (`PICK_GAP_ULPS`, 16 BF16 ulps) under its
//!    own, or above it -- ADR 0095's rule for the head, the verify tile's
//!    drift from a one-column decode. Each prompt runs through the engine
//!    (token ids, which the routes do not return) and through
//!    `/v1/chat/completions`, whose answer is the engine's run. BF16 KV, the
//!    correctness oracle (ADR 0022); prompt reuse off, so a prompt's prefill
//!    is never cut at a boundary a sibling published.
//! 2. **The counters keep their meaning.** Each route request's request-log
//!    line (`ignis.request.done`) reports the rounds, drafts and accepted
//!    drafts the engine's run of the same prompt reported; `/metrics`'
//!    `ignis_speculative_*` counters move by the sums of those lines; the
//!    window-0 load writes no `spec.*` field and exports no speculative
//!    series. The load's log names the drafter and the window.
//! 3. **Structured output.** A `json_schema` response format answers with a
//!    document valid under its schema; a request at the API's defaults
//!    (thinking on, sampled) drafts too.
//! 4. (after 5, on the same load) **Siblings over one system prompt reuse the shared prefix**, each
//!    answering its own question and drafting, seven of them claiming it at
//!    once, one a lane: every holder keeps its own copy of the prefix's last
//!    page (ADR 0095), on a pool at its floor, the pages admission keeps back
//!    for those copies included (GitHub #623), their reservations filling
//!    what admission holds. A pool a page under that floor is refused before
//!    anything loads, naming the copies.
//! 5. **A request evicted to KV-RAM goes on.** First on that floor pool, so no
//!    retained state stands in the room a move makes, an Agent
//!    request is decoding when an Interactive request reserving one whole
//!    context arrives: admission holds exactly that context beside the
//!    retained slots' pages, so the Agent request -- the eviction order's
//!    first victim -- is snapshotted to KV-RAM, restored once the intruder is
//!    done, and drafts again. It decodes alone on either side of its move, so
//!    it is held to the same request served alone on a fresh load of the same
//!    flags, token for token (GitHub #444).
//!
//! Its own test binary: the request log is written on the engine's telemetry
//! task, so the capture is the process's global subscriber.
//!
//! ```text
//! cargo test -p ignis-server --features cuda --test mtp_serving_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing
//! artifact or GPU is a skip, under it a failure. Four loads, one after the
//! other.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use ignis_artifact::{ChatMessage, ChatRenderOptions, FrontendSet, MessageContent, Reader, Role};
use ignis_core::compute::{ModelConfig, ModelFamily};
use ignis_core::types::{DecisionRead, DecodeParams, RequestClass, RequestInput, SchedEvent, SpecCounters};
use ignis_core::{gpu_profile, KvFormat, SpeculativeBackend};
use ignis_server::engine::{collect_readout, Engine};
use ignis_server::Server;

// ADR 0056's near-tie bound and one BF16 ulp, as the core tests' logit drift
// reads them: one copy of the rule.
#[path = "../../core/tests/support/logit_drift.rs"]
mod logit_drift;

use logit_drift::{bf16_ulp, PICK_GAP_ULPS};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// The MTP head's 12 objects (ADR 0095): what an MTP load's weights line
/// holds beyond a load without the head.
const HEAD_BYTES: u64 = 451_267_584;
/// The provisional default window (GitHub #628 sets it).
const WINDOW: u32 = 7;
/// Tokens each greedy prompt may generate.
const TOTAL: u32 = 128;
/// How long one request may take before the test calls it hung.
const DEADLINE: Duration = Duration::from_secs(600);
/// The floor pool's context: 64 pages.
const SMALL_CONTEXT: u32 = 4096;

const PROMPTS: [(&str, &str); 4] = [
    ("binary search", "Explain how a binary search works, step by step."),
    ("fibonacci", "Write a Python function that returns the n-th Fibonacci number, with a short docstring. Reply with the code only."),
    ("planets", "List the planets of the solar system, with one fact about each."),
    ("france", "In one word, what is the capital of France?"),
];

/// The greedy loads' flags: the operator's MTP choice, the 27B's vision off
/// (the head does not draft beside it until GitHub #624), BF16 KV, no prompt
/// reuse, a 16,384-token context on the card's whole budget.
const GREEDY: [&str; 14] = [
    "--spec-backend", "mtp", "--qwen38-vision-enabled", "false", "--model-kv-format", "bf16", "--reuse-prompt", "off",
    "--model-max-context", "16384", "--model-idle-unload", "0", "--model-loop-guard", "off",
];

fn flags(base: &[&str], extra: &[&str]) -> Vec<String> {
    base.iter().chain(extra).map(|f| f.to_string()).collect()
}

/// The 27B's options from `args`, as `main` resolves and fits them.
fn options(args: &[String]) -> ignis_server::config::Config {
    let ignis_server::config::ConfigOutcome::Config(config) =
        ignis_server::config::resolve(args, |_| None).unwrap_or_else(|e| panic!("resolve {args:?}: {e}"))
    else {
        panic!("expected a runnable config");
    };
    let config = config.for_family(ModelFamily::Qwen38_27b).unwrap_or_else(|e| panic!("the 27B from {args:?}: {e}"));
    ignis_server::config::Config { model: MODEL.to_owned(), model_named: true, ..config }
}

// ── the loads ───────────────────────────────────────────────────────────────

/// One served load: the model, the router, its metrics.
struct Load {
    model: ignis_server::ActiveModel,
    app: Router,
    metrics: Router,
}

impl Load {
    fn engine(&self) -> &Engine {
        self.model.engine().expect("a text model has an engine")
    }

    async fn close(self) {
        let Load { model, app, metrics } = self;
        drop((app, metrics));
        tokio::task::spawn_blocking(move || model.shut_down()).await.expect("the model shuts down");
    }
}

/// The 27B loaded from `args` through the server's load path, or `None`
/// through `skip_or_fail`.
fn load(args: &[String]) -> Option<Load> {
    let options = options(args);
    let loaded = match ignis_server::load::prepare_model(&options, Path::new(ARTIFACT)).and_then(ignis_server::load::load_model) {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the 27B from {args:?}: {e}"));
            return None;
        }
    };
    let server = Server::from_active(loaded.model.clone()).with_metrics().with_request_timeout(DEADLINE);
    let metrics = server.metrics_app().expect("metrics are on");
    Some(Load { model: loaded.model, app: server.app(), metrics })
}

// ── the request log ─────────────────────────────────────────────────────────

struct Log(Arc<ignis_logging::MemorySink>);

impl Log {
    fn events(&self) -> Vec<Value> {
        self.0.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect()
    }

    fn named(&self, name: &str) -> Vec<Value> {
        self.events().into_iter().filter(|event| event["event_name"] == name).collect()
    }

    /// The `name` lines once there are at least `count`: the telemetry task
    /// writes a request's after its response.
    async fn at_least(&self, name: &str, count: usize) -> Vec<Value> {
        let mut lines = Vec::new();
        for _ in 0..200 {
            lines = self.named(name);
            if lines.len() >= count {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(lines.len() >= count, "{count} `{name}` lines, {} logged", lines.len());
        lines
    }
}

/// A done line's speculative counters, `None` where it carries no `spec.*`.
fn logged_spec(done: &Value) -> Option<(u64, u64, u64)> {
    let a = &done["attributes"];
    Some((a["spec.rounds"].as_u64()?, a["spec.drafted"].as_u64()?, a["spec.accepted"].as_u64()?))
}

// ── requests ────────────────────────────────────────────────────────────────

fn chat_body(messages: Value, max_tokens: u32, class: &str) -> Value {
    json!({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": false,
        "enable_thinking": false,
        "class": class,
    })
}

fn question(text: &str, max_tokens: u32, class: &str) -> Value {
    chat_body(json!([{ "role": "user", "content": text }]), max_tokens, class)
}

/// A chat completion: the response, answered 200.
async fn chat(app: Router, body: Value) -> Value {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&bytes)));
    assert!(status.is_success(), "HTTP {status}: {value}");
    value
}

fn content(response: &Value) -> String {
    response.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// The sum of a Prometheus metric over its label sets, 0 while absent.
async fn metric(metrics: &Router, name: &str) -> u64 {
    let request = Request::builder().uri("/metrics").body(Body::empty()).unwrap();
    let response = metrics.clone().oneshot(request).await.unwrap();
    assert!(response.status().is_success(), "GET /metrics: {}", response.status());
    let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
    text.lines()
        .filter(|line| line.strip_prefix(name).is_some_and(|rest| rest.starts_with(' ') || rest.starts_with('{')))
        .map(|line| line.rsplit(' ').next().unwrap().parse::<f64>().unwrap() as u64)
        .sum()
}

async fn metrics_text(metrics: &Router) -> String {
    let request = Request::builder().uri("/metrics").body(Body::empty()).unwrap();
    let response = metrics.clone().oneshot(request).await.unwrap();
    String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap()
}

// ── the engine's runs ───────────────────────────────────────────────────────

/// `question` as the route renders it, thinking off: its tokens.
fn prompt_tokens(frontend: &FrontendSet, question: &str) -> Vec<u32> {
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Text(question.to_owned()),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let options = ChatRenderOptions { enable_thinking: false, ..Default::default() };
    let rendered = frontend.chat_template().render_with_thinking_and_tools(&messages, options, None).expect("render");
    frontend.tokenizer().encode(&rendered).expect("tokenize")
}

fn input(tokens: Vec<u32>, decision: Option<DecisionRead>) -> RequestInput {
    RequestInput {
        decision,
        model: MODEL.into(),
        tokens,
        params: DecodeParams { max_tokens: Some(TOTAL), ..DecodeParams::default() },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
    }
}

/// A greedy run: its tokens, and the speculative rounds it reported.
struct Run {
    tokens: Vec<u32>,
    spec: Option<SpecCounters>,
}

async fn run(engine: &Engine, name: &str, tokens: &[u32]) -> Run {
    let (_, mut events) = engine
        .submit(input(tokens.to_vec(), None), RequestClass::Interactive)
        .await
        .unwrap_or_else(|e| panic!("{name}: submit: {e:?}"));
    let mut out = Vec::new();
    loop {
        let event = tokio::time::timeout(DEADLINE, events.recv())
            .await
            .unwrap_or_else(|_| panic!("{name}: no event in {DEADLINE:?}"))
            .unwrap_or_else(|| panic!("{name}: the engine gave up"));
        match event {
            SchedEvent::Token { token, .. } => out.push(token),
            SchedEvent::Done { reason, spec, .. } => {
                assert!(!out.is_empty(), "{name}: nothing generated ({reason:?})");
                return Run { tokens: out, spec };
            }
            _ => {}
        }
    }
}

/// The logits of `picks` the prefill route of `load` gives after `prompt`
/// and `agreed`: a decision readout at that position.
async fn prefill_logits(engine: &Engine, name: &str, prompt: &[u32], agreed: &[u32], picks: [u32; 2]) -> [f32; 2] {
    let mut tokens = prompt.to_vec();
    tokens.extend_from_slice(agreed);
    let decision = DecisionRead::Answers(Arc::from(picks.to_vec()));
    let (_, mut events) = engine
        .submit(input(tokens, Some(decision)), RequestClass::Interactive)
        .await
        .unwrap_or_else(|e| panic!("{name}: submit the readout: {e:?}"));
    let readout = collect_readout(&mut events, DEADLINE).await.unwrap_or_else(|e| panic!("{name}: the readout: {e:?}"));
    [readout.logits[0], readout.logits[1]]
}

/// `on` (the MTP load's run) against `off` (the window-0 load's): identical,
/// or identical up to a first split where `off_engine`'s prefill route after
/// the agreed text puts `on`'s pick at most `PICK_GAP_ULPS` BF16 ulps under
/// `off`'s, or above it. A run that ended where the other goes on picked
/// `eos` there. Panics on a split past the near tie.
async fn judge(off_engine: &Engine, name: &str, prompt: &[u32], eos: u32, off: &[u32], on: &[u32]) -> Option<String> {
    let at = match off.iter().zip(on).position(|(a, b)| a != b) {
        Some(at) => at,
        None if off.len() == on.len() => return None,
        None => off.len().min(on.len()),
    };
    let pick = |run: &[u32]| run.get(at).copied().unwrap_or(eos);
    let (off_pick, on_pick) = (pick(off), pick(on));
    let [off_logit, on_logit] = prefill_logits(off_engine, name, prompt, &off[..at], [off_pick, on_pick]).await;
    let gap = off_logit - on_logit;
    let ulp = bf16_ulp(off_logit.abs().max(on_logit.abs()));
    let ulps = gap / ulp;
    let line = format!(
        "{name}: MTP splits from spec off at token {at}: off {off_pick} on {on_pick}; the prefill route's gap off - on = \
         {gap} ({ulps} ulps of {ulp}): the near tie ({PICK_GAP_ULPS} ulps) {}",
        if ulps <= PICK_GAP_ULPS { "holds" } else { "FAILS" }
    );
    println!("{line}");
    assert!(ulps <= PICK_GAP_ULPS, "{line}: the prefill route sides with spec off past the near tie");
    Some(line)
}

fn plan_line(log: &[Value]) -> Value {
    let plans: Vec<&Value> = log.iter().filter(|e| e["event_name"] == "ignis.runtime.vram_plan").collect();
    assert_eq!(plans.len(), 1, "one plan per load");
    plans[0]["attributes"].clone()
}

fn loaded_line(log: &[Value]) -> Value {
    let loaded: Vec<&Value> = log.iter().filter(|e| e["event_name"] == "ignis.model.loaded").collect();
    assert_eq!(loaded.len(), 1, "one load line per load");
    loaded[0]["attributes"].clone()
}

/// The pages admission keeps back on an MTP load (GitHub #622): one a lane
/// and one a retained slot, as the plan's line states them.
fn holder_pages(plan: &Value, lanes: u64) -> u64 {
    lanes + plan["retained_device_slots"].as_u64().unwrap() + plan["retained_host_slots"].as_u64().unwrap()
}

/// A system prompt long enough to publish a prefix of several pages.
fn system_prompt() -> String {
    let mut text = String::from("You are a concise assistant inside an automated quiz. Follow every rule below.\n");
    for rule in 1..=40 {
        text.push_str(&format!("Rule {rule}: answer the question asked, in as few words as it needs, with no preamble.\n"));
    }
    text
}

/// The Agent request the KV-RAM leg moves: a few hundred tokens, and a
/// reservation of 24 pages, so it and a whole context exceed what admission
/// holds on the floor pool.
fn agent_request() -> Value {
    question("Write the English words for the numbers one through one hundred, separated by commas.", 1500, "agent")
}

// ── the test ────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn an_mtp_load_serves_text_structured_output_shared_prefixes_and_kv_ram() {
    if !Path::new(ARTIFACT).exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let sink = Arc::new(ignis_logging::MemorySink::new());
    tracing_subscriber::registry()
        .with(ignis_logging::JsonLayer::new(sink.clone()))
        .try_init()
        .expect("this test binary installs the only subscriber");
    let log = Log(sink);
    let frontend = FrontendSet::from_reader(&Reader::open(Path::new(ARTIFACT)).expect("open")).expect("frontend");
    let eos = frontend.eos_token_id().expect("eos");
    let prompts: Vec<(&str, Vec<u32>)> = PROMPTS.iter().map(|(name, q)| (*name, prompt_tokens(&frontend, q))).collect();

    // ── 1-3. MTP at its default window, the card's whole budget ─────────────
    let mark = log.events().len();
    let Some(on) = load(&flags(&GREEDY, &[])) else { return };
    let loaded_log = log.events().split_off(mark);
    let loaded = loaded_line(&loaded_log);
    println!("MTP load: {loaded}");
    assert_eq!(loaded["speculation"], format!("mtp draft_tokens={WINDOW}"), "the log names the drafter and the window");
    let lanes = loaded["decode_lanes"].as_u64().unwrap();
    let on_plan = plan_line(&loaded_log);
    println!("MTP plan: {on_plan}");
    // The bound (GitHub #622, #623): admission keeps back a page a lane and
    // a page a retained slot, and the pool holds them beside one context and
    // the retained slots' pages, so the ledger never asks the leaf for a page
    // past its pool.
    let reserve = holder_pages(&on_plan, lanes);
    assert_eq!(on_plan["kv_holder_pages"].as_u64(), Some(reserve), "{on_plan}");
    let retained = reserve - lanes;
    let floor = 16_384 / 64 + retained + reserve;
    assert!(on_plan["kv_pool_pages"].as_u64().unwrap() >= floor, "the served pool holds its floor: {on_plan}");
    assert_eq!(on_plan["drafter_round_bytes"], 0, "no DFlash2 round");

    let done_at_load = log.named("ignis.request.done").len();
    let mut on_runs = Vec::new();
    for (name, tokens) in &prompts {
        let run = run(on.engine(), name, tokens).await;
        let spec = run.spec.unwrap_or_else(|| panic!("{name}: the MTP load ran no verify round"));
        println!("{name}: {} tokens, {} rounds, {} drafted, {} accepted", run.tokens.len(), spec.rounds, spec.drafted, spec.accepted);
        on_runs.push(run);
    }
    // The engine's runs are requests too: their lines and their counters
    // first, the sums of what the runs reported.
    let engine_sum = on_runs.iter().filter_map(|run| run.spec).fold((0, 0, 0), |sum, spec| {
        (sum.0 + u64::from(spec.rounds), sum.1 + u64::from(spec.drafted), sum.2 + u64::from(spec.accepted))
    });
    log.at_least("ignis.request.done", done_at_load + prompts.len()).await;
    let mut counted = (0, 0, 0);
    for _ in 0..200 {
        counted = (
            metric(&on.metrics, "ignis_speculative_rounds_total").await,
            metric(&on.metrics, "ignis_speculative_drafted_tokens_total").await,
            metric(&on.metrics, "ignis_speculative_accepted_tokens_total").await,
        );
        if counted == engine_sum {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(counted, engine_sum, "/metrics counts the engine's runs' rounds, drafts and accepted drafts");
    // Through the route: the engine's text, and the request log's counters
    // are the engine run's.
    let (rounds_before, drafted_before, accepted_before) = (
        metric(&on.metrics, "ignis_speculative_rounds_total").await,
        metric(&on.metrics, "ignis_speculative_drafted_tokens_total").await,
        metric(&on.metrics, "ignis_speculative_accepted_tokens_total").await,
    );
    let positions_before = metric(&on.metrics, "ignis_speculative_position_accepted_total").await;
    let done_before = log.named("ignis.request.done").len();
    let mut logged_sum = (0, 0, 0);
    for (index, ((name, question_text), run)) in PROMPTS.iter().zip(&on_runs).enumerate() {
        let response = chat(on.app.clone(), question(question_text, TOTAL, "interactive")).await;
        let mut ids = run.tokens.clone();
        if ids.last() == Some(&eos) {
            ids.pop();
        }
        let engine_text = frontend.tokenizer().decode(&ids).expect("decode");
        assert_eq!(content(&response).trim(), engine_text.trim(), "{name}: the route's answer is the engine's run");
        let done = log.at_least("ignis.request.done", done_before + index + 1).await;
        let line = &done[done_before + index];
        let spec = run.spec.expect("drafted above");
        let logged = logged_spec(line).unwrap_or_else(|| panic!("{name}: no spec.* on the MTP load's line: {line}"));
        assert_eq!(logged, (u64::from(spec.rounds), u64::from(spec.drafted), u64::from(spec.accepted)), "{name}: {line}");
        assert!(logged.0 > 0 && logged.1 <= u64::from(WINDOW) * logged.0 && logged.2 <= logged.1, "{name}: {line}");
        let completion = response.pointer("/usage/completion_tokens").and_then(Value::as_u64).unwrap();
        let tokens = line["attributes"]["tokens"].as_u64().unwrap();
        assert!((completion..=completion + 1).contains(&tokens), "{name}: the line's {tokens} tokens against the usage's {completion}");
        assert!(completion <= logged.0 + logged.2 + 1, "{name}: a round commits its accepted drafts and one token: {line}");
        logged_sum = (logged_sum.0 + logged.0, logged_sum.1 + logged.1, logged_sum.2 + logged.2);
    }
    let mut moved = (0, 0, 0);
    for _ in 0..100 {
        moved = (
            metric(&on.metrics, "ignis_speculative_rounds_total").await - rounds_before,
            metric(&on.metrics, "ignis_speculative_drafted_tokens_total").await - drafted_before,
            metric(&on.metrics, "ignis_speculative_accepted_tokens_total").await - accepted_before,
        );
        if moved == logged_sum {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(moved, logged_sum, "/metrics moved by the request log's counters");
    let positions = metric(&on.metrics, "ignis_speculative_position_accepted_total").await - positions_before;
    assert_eq!(positions, logged_sum.2, "a draft committed at a position counts once at it");

    // Structured output: a document valid under its schema.
    let schema_text = r#"{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}},"required":["city","population"],"additionalProperties":false}"#;
    let mut body = question("Give the capital of Italy and its population as JSON.", 96, "interactive");
    body["response_format"] = json!({ "type": "json_schema", "json_schema": { "name": "city", "strict": true, "schema": serde_json::from_str::<Value>(schema_text).unwrap() } });
    let response = chat(on.app.clone(), body).await;
    let answer = content(&response);
    println!("structured output on the MTP load: {answer}");
    let schema = ignis_grammar::json::Json::parse(schema_text).expect("the schema");
    let document = ignis_grammar::json::Json::parse(&answer).unwrap_or_else(|e| panic!("the answer is not JSON ({e}): {answer}"));
    let faults = ignis_grammar::validate::violations(&schema, &document);
    assert!(faults.is_empty(), "the answer {answer}: {faults:?}");
    let done = log.at_least("ignis.request.done", done_before + PROMPTS.len() + 1).await;
    println!("structured output's request log: {}", done[done_before + PROMPTS.len()]);
    // A request at the API's defaults: thinking on, sampled at the model's
    // own settings, so the head drafts for a sampling lane and the reasoning
    // closes inside its rounds.
    let defaults = json!({ "model": MODEL, "messages": [{ "role": "user", "content": "Why is the sky blue? Two sentences." }], "max_tokens": 400 });
    let response = chat(on.app.clone(), defaults).await;
    let message = &response["choices"][0]["message"];
    let said = format!("{}{}", message["reasoning_content"].as_str().unwrap_or_default(), message["content"].as_str().unwrap_or_default());
    assert!(!said.trim().is_empty(), "a defaults request says something: {response}");
    let done = log.at_least("ignis.request.done", done_before + PROMPTS.len() + 2).await;
    let line = &done[done_before + PROMPTS.len() + 1];
    println!("a request at the defaults: {line}");
    assert!(logged_spec(line).is_some_and(|(rounds, _, _)| rounds > 0), "the head drafted for a sampling lane: {line}");
    on.close().await;

    // ── 1-2. The same flags at a window of 0: no speculation ────────────────
    let mark = log.events().len();
    let Some(off) = load(&flags(&GREEDY, &["--spec-draft-tokens", "0"])) else { return };
    let loaded_log = log.events().split_off(mark);
    let loaded = loaded_line(&loaded_log);
    assert_eq!(loaded["speculation"], "off (mtp draft_tokens=0)", "{loaded}");
    let off_plan = plan_line(&loaded_log);
    println!("window-0 plan: {off_plan}");
    assert_eq!(off_plan["kv_holder_pages"], 0, "no head, no holder pages");
    assert_eq!(
        on_plan["weights_bytes"].as_u64().unwrap() - off_plan["weights_bytes"].as_u64().unwrap(),
        HEAD_BYTES,
        "nothing of the head bound at a window of 0"
    );
    let mut splits = Vec::new();
    for ((name, tokens), on_run) in prompts.iter().zip(&on_runs) {
        let off_run = run(off.engine(), name, tokens).await;
        assert!(off_run.spec.is_none(), "{name}: the window-0 load ran verify rounds");
        if let Some(line) = judge(off.engine(), name, tokens, eos, &off_run.tokens, &on_run.tokens).await {
            splits.push(line);
        }
    }
    let done_before = log.named("ignis.request.done").len();
    chat(off.app.clone(), question(PROMPTS[3].1, TOTAL, "interactive")).await;
    let done = log.at_least("ignis.request.done", done_before + 1).await;
    assert!(logged_spec(&done[done_before]).is_none(), "no spec.* field without speculation: {}", done[done_before]);
    assert!(!metrics_text(&off.metrics).await.contains("ignis_speculative_rounds_total"), "no speculative series");
    off.close().await;
    println!("{} of {} prompts split at a near tie", splits.len(), prompts.len());

    // ── 4-5. A pool at its floor, prompt reuse on, hq-e8-2b ─────────────────
    // The lanes named, so a plan that refuses does not adapt them away.
    let small = [
        "--spec-backend", "mtp", "--qwen38-vision-enabled", "false", "--model-max-context", "4096", "--spec-decode-lanes", "8",
        "--model-idle-unload", "0", "--model-loop-guard", "off",
    ];
    let unsized_options = options(&flags(&small, &[]));
    let shape = ignis_server::runtime::EngineShape::from(&unsized_options);
    let lanes = shape.decode_lanes_for(ModelFamily::Qwen38_27b);
    let retained = shape.retained_slots();
    let reserve = ignis_core::seq::holder_page_reserve(&ModelConfig::qwen38_27b(), Some(SpeculativeBackend::Mtp), lanes, retained);
    assert_eq!(reserve, lanes + retained);
    let floor_pages = SMALL_CONTEXT / 64 + retained + reserve;
    let page_bytes = KvFormat::HqE8_2b.page_bytes(ignis_core::seq::kv_geometry(&ModelConfig::qwen38_27b(), Some(SpeculativeBackend::Mtp)));
    let pool = (u64::from(floor_pages) * page_bytes).to_string();
    let floor_flags = flags(&small, &["--vram-kv-pool-bytes", &pool]);
    // A page under it is refused before anything loads, naming the copies.
    let under = (u64::from(floor_pages - 1) * page_bytes).to_string();
    let refused = ignis_server::load::prepare_model(&options(&flags(&small, &["--vram-kv-pool-bytes", &under])), Path::new(ARTIFACT))
        .and_then(ignis_server::load::load_model)
        .err()
        .expect("a pool a page under the floor is refused")
        .to_string();
    assert!(refused.contains("the MTP head's holders copy"), "{refused}");
    let mark = log.events().len();
    let Some(tight) = load(&floor_flags) else { return };
    let tight_plan = plan_line(&log.events().split_off(mark));
    println!("floor plan: {tight_plan}");
    assert_eq!(tight_plan["kv_pool_pages"].as_u64(), Some(u64::from(floor_pages)), "the pool at its floor");
    assert_eq!(tight_plan["kv_holder_pages"].as_u64(), Some(u64::from(reserve)));
    assert_eq!(holder_pages(&tight_plan, u64::from(lanes)), u64::from(reserve), "the load's lanes and retained slots");

    // An Agent request through KV-RAM for an Interactive one reserving the
    // whole context: admission holds it only because the floor held the
    // holders' pages beside it.
    // First on the load, so no retained state stands in the room the move
    // makes: the scheduler moves a victim only when that makes the whole room.
    let ttft_before = log.named("ignis.request.ttft").len();
    let done_before = log.named("ignis.request.done").len();
    let agent = tokio::spawn(chat(tight.app.clone(), agent_request()));
    log.at_least("ignis.request.ttft", ttft_before + 1).await;
    println!("pages in use as the intruder arrives: {}", metric(&tight.metrics, "ignis_kv_pool_used_pages").await);
    let intruder = chat(tight.app.clone(), question("What is 2 + 2? Answer with a number only.", SMALL_CONTEXT - 64, "interactive")).await;
    let agent = agent.await.expect("the agent request");
    assert!(content(&intruder).contains('4'), "{intruder}");
    let moved_text = content(&agent);
    println!("moved agent request: {moved_text:?}");
    assert!(moved_text.to_lowercase().contains("hundred"), "restored, it went on to the end: {moved_text:?}");
    let done = log.at_least("ignis.request.done", done_before + 2).await;
    let agents: Vec<u64> = done[done_before..]
        .iter()
        .filter(|e| e["attributes"]["class"] == "agent")
        .filter_map(|e| e["attributes"]["request_id"].as_u64())
        .collect();
    assert_eq!(agents.len(), 1, "one Agent request");
    let ids = |name: &str| log.named(name).iter().filter_map(|e| e["attributes"]["request_id"].as_u64()).collect::<Vec<_>>();
    assert_eq!(ids("ignis.request.evicted"), agents, "the Agent request is the one evicted");
    assert_eq!(ids("ignis.request.restored"), agents, "and restored from its blob, not prefilled again");
    let agent_line = done.iter().find(|e| e["attributes"]["request_id"].as_u64() == Some(agents[0])).unwrap();
    println!("moved agent request's line: {agent_line}");
    assert!(logged_spec(agent_line).is_some_and(|(_, drafted, accepted)| drafted > 0 && accepted > 0), "it drafted: {agent_line}");

    // Siblings over one system prompt: the first publishes it, seven claim it
    // at once, every lane holding a copy of its last page, their reservations
    // filling what admission holds.
    let system = system_prompt();
    let sibling = |q: &str, max_tokens: u32| {
        chat_body(json!([{ "role": "system", "content": system }, { "role": "user", "content": q }]), max_tokens, "interactive")
    };
    let reused = |metrics: Router| async move {
        metric(&metrics, "ignis_prefix_reused_tokens_total").await + metric(&metrics, "ignis_retained_reused_tokens_total").await
    };
    let done_before = log.named("ignis.request.done").len();
    let first = chat(tight.app.clone(), sibling("What is the capital of Italy? Answer in one word.", 32)).await;
    assert!(content(&first).contains("Rome"), "{first}");
    let reused_before = reused(tight.metrics.clone()).await;
    let claimants = [
        ("What is the capital of Japan? Answer in one word.", "Tokyo"),
        ("What is the capital of France? Answer in one word.", "Paris"),
        ("What is twelve times twelve? Answer with the number only.", "144"),
        ("Which is the largest planet of the solar system? Answer in one word.", "Jupiter"),
        ("What is the chemical symbol for gold? Answer with the symbol only.", "Au"),
        ("Who wrote Hamlet? Answer with the surname only.", "Shakespeare"),
        ("What colour is a clear daytime sky? Answer in one word.", "lue"),
    ];
    let answers: Vec<_> = claimants.iter().map(|(q, _)| tokio::spawn(chat(tight.app.clone(), sibling(q, 400)))).collect();
    let mut said = Vec::new();
    for answer in answers {
        said.push(content(&answer.await.expect("a claimant")));
    }
    let done = log.at_least("ignis.request.done", done_before + 1 + claimants.len()).await;
    let reused_tokens = reused(tight.metrics.clone()).await - reused_before;
    println!("claimants: {said:?}; reused {reused_tokens} prompt tokens");
    for ((q, expected), answer) in claimants.iter().zip(&said) {
        assert!(answer.contains(expected), "{q}: {answer:?}");
    }
    assert!(reused_tokens > 0, "the siblings reused the shared prefix");
    for line in &done[done_before..done_before + 1 + claimants.len()] {
        println!("sibling: {line}");
        assert!(logged_spec(line).is_some_and(|(rounds, _, _)| rounds > 0), "a sibling ran verify rounds: {line}");
    }

    tight.close().await;

    // The same request alone on a fresh load of the same flags.
    let Some(fresh) = load(&floor_flags) else { return };
    let unmoved = chat(fresh.app.clone(), agent_request()).await;
    assert_eq!(content(&unmoved), moved_text, "the moved request generated what it generates unmoved");
    fresh.close().await;
}
