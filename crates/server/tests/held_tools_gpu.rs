//! Held tools on the served 27B (GitHub #611 and #617, spec #582, ADR 0094):
//! chat completions and the Responses API with **strict tools**, and with
//! the Responses default (a tool sent without `strict`, normalized, #614),
//! through the real router over the real GPU-backed scheduler with DFlash2
//! on, the server's own grammars compiled against the artifact's tokenizer
//! (`ArtifactTemplateProvider::serving`).
//!
//! **The corpus** (`support/held_tools_corpus.rs`): agent requests whose
//! tools are held -- free strings, enums, a pattern and a format, integers
//! with bounds, a boolean, nested objects and an array of them, nullable
//! parameters, a tool of nine parameters; long values (a script, a file
//! about the call dialect whose content holds the dialect's own tags, a
//! typed value of ten edits); parallel calls; a free tool beside held ones;
//! `tool_choice` `"required"` and named; `response_format` beside the tools;
//! thinking on and off; the Responses API with strict tools and with its
//! default -- and cases built against the model's prior, so that it misses:
//! a value the prompt puts outside an enum, a bound or a pattern, a word
//! where a number goes, a day that is not a date, required parameters the
//! prompt gives nothing for or whose description calls them optional (the
//! early close), parameters the tool does not declare. Every call names a
//! declared tool and every held call's arguments are valid under its tool's
//! `parameters` (#608's validator; under the Responses default, valid under
//! the schema as sent and no key it does not declare); every request is
//! counted as held, every miss repaired, and the verify rounds keep their
//! drafts.
//!
//! **The repairs, read.** The server shows no repair's text: a miss is
//! counted (`ignis_structured_misses_total`), not logged. So each request is
//! served beside its **free twin**, the same request with every tool sent
//! `strict: false`, and both deliveries are printed: the twin's calls with
//! each fault the validator finds in them and, beside each, what the held
//! call wrote at the same place -- the early close's above all (a required
//! parameter the twin left out, the held call wrote). On the Responses
//! default the twin's prompt is the request's own (the template renders a
//! tool sent without `strict` as `strict: false`), so greedy, the twin is
//! what the model wrote before the first miss; on chat completions the
//! prompts differ by that one word. Where the two deliveries part is
//! printed too, and both prompts' token counts.
//!
//! **The cost on held traffic**, per request and pooled: each request's
//! tokens, its decode time (its duration less its time to first token, off
//! the metrics' sums), its verify rounds, drafted and accepted tokens, held
//! beside free. The two do different work past the first miss, so the
//! ratio is a reading, not a gate.
//!
//! **One call per turn**: `parallel_tool_calls: false` on chat completions
//! and the Responses API, held tools and free, against a control the same
//! conversation without the field, which makes two calls or more in its
//! first turn.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile it is a failure.
//! Each test loads the 27B: run them one at a time.
//!
//! ```text
//! cargo test -p ignis-server --features cuda --test held_tools_gpu -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

#[path = "support/held_tools_corpus.rs"]
mod corpus;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::Value as JsonValue;
use tower::ServiceExt;

use corpus::{Api, Case, Delivered, Expect, Hold};
use ignis_artifact::{FrontendSet, Reader};
use ignis_core::compute::ModelFamily;
use ignis_core::{gpu_profile, Speculation, SpeculativeBackend};
use ignis_server::engine::Engine;
use ignis_server::runtime::{cuda_scheduler, EngineShape};
use ignis_server::telemetry::SystemClock;
use ignis_server::Server;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
/// DFlash2-7, every built-in profile's window.
const WINDOW: u32 = 7;

/// A live server over the real scheduler with DFlash2, its grammars the
/// load's own; the router dropped before the model thread is joined, so the
/// next load never allocates beside this one.
struct Harness {
    server: Option<Server>,
    driver: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(self.server.take());
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

fn harness() -> Option<Harness> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));
    let shape = EngineShape {
        speculation: Some(Speculation::new(SpeculativeBackend::Dflash2, WINDOW).expect("dflash2-7")),
        ..EngineShape::default()
    };
    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
        Ok((scheduler, _reservations)) => scheduler,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let provider = ignis_server::artifact_template::ArtifactTemplateProvider::serving(frontend, ModelFamily::Qwen38_27b, ignis_grammar::GrammarLimits::default());
    let server = Server::new(engine, Box::new(provider)).with_metrics().with_request_timeout(Duration::from_secs(600));
    Some(Harness { server: Some(server), driver: Some(driver) })
}

/// `body` posted to `path`: the status and the body back.
async fn send(server: &Server, path: &'static str, body: String) -> (u16, String) {
    let request = Request::builder().method("POST").uri(path).header("content-type", "application/json").body(Body::from(body)).unwrap();
    let response = server.app().oneshot(request).await.expect("the router answers");
    let status = response.status().as_u16();
    (status, String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap())
}

/// A series' value in the exposition, 0 when absent.
fn value(exposition: &str, series: &str) -> f64 {
    exposition.lines().find_map(|line| line.strip_prefix(series)?.strip_prefix(' ')?.parse().ok()).unwrap_or(0.0)
}

/// What the metrics say: read whole, or as the difference across one
/// request.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    tool_requests: f64,
    json_requests: f64,
    misses: f64,
    repairs: f64,
    rounds: f64,
    drafted: f64,
    accepted: f64,
    tokens: f64,
    ttft_s: f64,
    duration_s: f64,
    valid: f64,
    invalid: f64,
    dropped: f64,
}

impl Counts {
    fn read(exposition: &str) -> Self {
        let v = |series: &str| value(exposition, series);
        let both = |family: &str| v(&format!("{family}{{type=\"tool\"}}")) + v(&format!("{family}{{type=\"json_schema\"}}"));
        Counts {
            tool_requests: v("ignis_structured_requests_total{type=\"tool\"}"),
            json_requests: v("ignis_structured_requests_total{type=\"json_schema\"}"),
            misses: both("ignis_structured_misses_total"),
            repairs: both("ignis_structured_repairs_total"),
            rounds: v("ignis_speculative_rounds_total"),
            drafted: v("ignis_speculative_drafted_tokens_total"),
            accepted: v("ignis_speculative_accepted_tokens_total"),
            tokens: v("ignis_generated_tokens_total"),
            ttft_s: v("ignis_request_ttft_seconds_sum"),
            duration_s: v("ignis_request_duration_seconds_sum"),
            valid: v("ignis_tool_calls_total{outcome=\"valid\"}"),
            invalid: v("ignis_tool_calls_total{outcome=\"invalid\"}"),
            dropped: v("ignis_tool_calls_total{outcome=\"dropped\"}"),
        }
    }

    fn since(self, before: Counts) -> Self {
        Counts {
            tool_requests: self.tool_requests - before.tool_requests,
            json_requests: self.json_requests - before.json_requests,
            misses: self.misses - before.misses,
            repairs: self.repairs - before.repairs,
            rounds: self.rounds - before.rounds,
            drafted: self.drafted - before.drafted,
            accepted: self.accepted - before.accepted,
            tokens: self.tokens - before.tokens,
            ttft_s: self.ttft_s - before.ttft_s,
            duration_s: self.duration_s - before.duration_s,
            valid: self.valid - before.valid,
            invalid: self.invalid - before.invalid,
            dropped: self.dropped - before.dropped,
        }
    }

    /// Decode seconds: from the first token to the end.
    fn decode_s(&self) -> f64 {
        (self.duration_s - self.ttft_s).max(0.0)
    }

    /// Tokens after the first over the decode seconds, as the bench's
    /// replay reads a request.
    fn decode_rate(&self) -> f64 {
        match self.decode_s() > 0.0 && self.tokens > 1.0 {
            true => (self.tokens - 1.0) / self.decode_s(),
            false => 0.0,
        }
    }
}

const DURATIONS: &str = "ignis_request_duration_seconds_count";

/// The exposition once `requests` requests are recorded: a request's
/// duration is the last thing its end records, after its response is sent,
/// so a read straight after the response races.
async fn recorded(metrics: &ignis_server::metrics::Metrics, requests: u64) -> String {
    for _ in 0..3_000 {
        let exposition = metrics.render();
        if value(&exposition, DURATIONS) as u64 >= requests {
            return exposition;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{requests} requests never recorded:\n{}", metrics.render());
}

/// One request served: what it delivered, what the metrics say it did.
struct Served {
    delivered: Delivered,
    counts: Counts,
}

/// `case` served, or (`held` false) its free twin; `done` requests served
/// before it.
async fn serve(server: &Server, case: &Case, held: bool, done: &mut u64) -> Served {
    let metrics = server.metrics.as_ref().expect("metrics on");
    let before = recorded(metrics, *done).await;
    let (status, text) = send(server, case.api.path(), case.body(MODEL, held, case.max_tokens())).await;
    assert_eq!(status, 200, "{} (held {held}): {text}", case.label);
    *done += 1;
    let after = recorded(metrics, *done).await;
    let response: JsonValue = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
    Served { delivered: Delivered::read(case.api, &response), counts: Counts::read(&after).since(Counts::read(&before)) }
}

/// How a turn that calls ends.
fn ends_calling(api: Api) -> &'static str {
    match api {
        Api::Chat => "tool_calls",
        Api::Responses => "completed",
    }
}

/// The held request keeps the promise: what its case expects, every call to
/// a declared tool, every held call valid under its tool's schema as sent,
/// each tool sent without `strict` echoed held, the request counted as held.
fn check_held(case: &Case, served: &Served) {
    let what = format!("{} ({:?})", case.label, case.api);
    let tools = corpus::declared(case.api, &case.tools_sent(true));
    let delivered = &served.delivered;
    let calls = &delivered.calls;
    match case.expect {
        Expect::Call => assert!(!calls.is_empty(), "{what}: no call: {delivered:?}"),
        Expect::Named(name) => {
            assert!(!calls.is_empty(), "{what}: no call: {delivered:?}");
            assert!(calls.iter().all(|call| call.name == name), "{what}: a call to another tool than {name}: {calls:?}");
        }
        Expect::CallOrAnswer if calls.is_empty() => {
            let schema = case.answer_schema().expect("the case's response_format");
            let answer = ignis_grammar::json::Json::parse(&delivered.text).unwrap_or_else(|e| panic!("{what}: the answer is not JSON ({e}): {delivered:?}"));
            let faults = ignis_grammar::validate::violations(&schema, &answer);
            assert!(faults.is_empty(), "{what}: the answer {:?}: {faults:?}", delivered.text);
        }
        Expect::CallOrAnswer => {}
    }
    if !calls.is_empty() {
        assert_eq!(delivered.end, ends_calling(case.api), "{what}: {delivered:?}");
    }
    for call in calls {
        let tool = tools.iter().find(|tool| tool.name == call.name).unwrap_or_else(|| panic!("{what}: a call to {}, which the request does not declare", call.name));
        if tool.hold != Hold::Free {
            let faults = corpus::faults(&tool.parameters, &call.arguments).unwrap_or_else(|| panic!("{what}: {}'s arguments are not JSON: {}", call.name, call.arguments));
            assert!(faults.is_empty(), "{what}: {} {}: {faults:?}", call.name, call.arguments);
        }
    }
    if case.api == Api::Responses {
        for (tool, echoed) in tools.iter().zip(&delivered.echoed_strict) {
            if tool.hold == Hold::Normalized {
                assert_eq!(echoed, &JsonValue::Bool(true), "{what}: {} sent without strict is echoed held", tool.name);
            }
        }
    }
    let counted = match case.counted_as() {
        "tool" => served.counts.tool_requests,
        _ => served.counts.json_requests,
    };
    assert_eq!(counted, 1.0, "{what}: counted once under type=\"{}\": {:?}", case.counted_as(), served.counts);
}

/// One fault of a free twin's call beside what the held call wrote there.
struct Read {
    case: String,
    bait: &'static str,
    fault: corpus::Fault,
    free: String,
    held: String,
}

fn side(name: &str, served: &Served) {
    let (c, d) = (&served.counts, &served.delivered);
    eprintln!(
        "  {name}: {} · {} calls · {} misses, {} repaired · {} prompt + {} tokens · {} rounds, {} drafted, {} accepted · decode {:.1} tok/s · the server's check: {} valid, {} invalid, {} dropped",
        d.end,
        d.calls.len(),
        c.misses,
        c.repairs,
        d.prompt_tokens,
        d.completion_tokens,
        c.rounds,
        c.drafted,
        c.accepted,
        c.decode_rate(),
        c.valid,
        c.invalid,
        c.dropped
    );
    if !d.reasoning.is_empty() {
        eprintln!("    reasoning: {} characters", d.reasoning.chars().count());
    }
    if !d.text.is_empty() {
        eprintln!("    text: {:?}", d.text);
    }
}

/// Print a case's two deliveries, the faults of the twin's calls beside the
/// held calls' values there, and where the two part; the faults read.
fn print_pair(case: &Case, held: &Served, free: &Served) -> Vec<Read> {
    let bait = if case.bait.is_empty() { String::new() } else { format!(", built to miss: {}", case.bait) };
    let field = if case.field.is_empty() { String::new() } else { format!(", {}", case.field) };
    let default = if case.without_strict { ", sent without strict" } else { "" };
    eprintln!("\n── {} [{:?}{default}{}{bait}{field}]", case.label, case.api, if case.thinking { ", thinking on" } else { "" });
    eprintln!("  prompt: {}", case.prompt);
    side("held", held);
    for call in &held.delivered.calls {
        eprintln!("    {} {}", call.name, call.arguments);
    }
    side("free", free);
    let tools = corpus::declared(case.api, &case.tools_sent(false));
    let mut read = Vec::new();
    for (i, call) in free.delivered.calls.iter().enumerate() {
        eprintln!("    {} {}", call.name, call.arguments);
        let Some(tool) = tools.iter().find(|tool| tool.name == call.name) else {
            eprintln!("      not a declared tool");
            continue;
        };
        let Some(faults) = corpus::faults(&tool.parameters, &call.arguments) else {
            eprintln!("      its arguments are not JSON");
            continue;
        };
        let calls = &held.delivered.calls;
        let beside = calls.get(i).filter(|held| held.name == call.name).or_else(|| calls.iter().find(|held| held.name == call.name));
        for fault in faults {
            let free_value = corpus::shown_at(&call.arguments, &fault.at);
            let held_value = beside.map_or("(no such call)".into(), |held| corpus::shown_at(&held.arguments, &fault.at));
            eprintln!("      {} at {} ({}): free wrote {free_value}, held wrote {held_value}", fault.kind, fault.at, fault.keyword);
            read.push(Read { case: format!("{} ({:?})", case.label, case.api), bait: case.bait, fault, free: free_value, held: held_value });
        }
    }
    let (prompts, same) = ((held.delivered.prompt_tokens, free.delivered.prompt_tokens), held.delivered.prompt_tokens == free.delivered.prompt_tokens);
    match (case.api, case.without_strict) {
        (Api::Responses, true) => eprintln!("  prompts {prompts:?} tokens: {}", if same { "the same length, as the twin's prompt is the request's own" } else { "NOT the same length" }),
        _ => eprintln!("  prompts {prompts:?} tokens (the template renders each tool's strict)"),
    }
    let (a, b) = (held.delivered.delivery(), free.delivered.delivery());
    match corpus::parting(&a, &b) {
        None => eprintln!("  the two deliveries are the same"),
        Some(at) => eprintln!("  they part at byte {at}: held «{}» free «{}»", corpus::snippet(&a, at, 60), corpus::snippet(&b, at, 60)),
    }
    read
}

/// Requests pooled: their tokens over their decode seconds.
#[derive(Default)]
struct Pool {
    requests: f64,
    tokens: f64,
    decode_s: f64,
    rounds: f64,
    drafted: f64,
    accepted: f64,
    misses: f64,
}

impl Pool {
    fn add(&mut self, counts: &Counts) {
        self.requests += 1.0;
        self.tokens += counts.tokens;
        self.decode_s += counts.decode_s();
        self.rounds += counts.rounds;
        self.drafted += counts.drafted;
        self.accepted += counts.accepted;
        self.misses += counts.misses;
    }

    fn rate(&self) -> f64 {
        match self.decode_s > 0.0 {
            true => (self.tokens - self.requests) / self.decode_s,
            false => 0.0,
        }
    }

    fn line(&self, name: &str) -> String {
        format!(
            "{name}: {} requests, {} tokens in {:.2} s of decode: {:.1} tok/s; {} rounds ({:.3} a token), {:.1}% of {} drafts accepted; {} misses",
            self.requests,
            self.tokens,
            self.decode_s,
            self.rate(),
            self.rounds,
            if self.tokens > 0.0 { self.rounds / self.tokens } else { 0.0 },
            if self.drafted > 0.0 { 100.0 * self.accepted / self.drafted } else { 0.0 },
            self.drafted,
            self.misses
        )
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn strict_tools_on_the_27b_with_dflash2_return_valid_calls_their_repairs_counted_and_drafts_kept() {
    let Some(harness) = harness() else { return };
    let server = harness.server.as_ref().unwrap();
    let mut done = 0;
    let (mut held_pool, mut free_pool) = (BTreeMap::<String, Pool>::new(), BTreeMap::<String, Pool>::new());
    let mut baits = BTreeMap::<&str, (u32, f64)>::new();
    let mut read = Vec::new();
    let (mut misses, mut repairs, mut drafted, mut accepted) = (0.0, 0.0, 0.0, 0.0);
    for case in &corpus::corpus() {
        let held = serve(server, case, true, &mut done).await;
        let free = serve(server, case, false, &mut done).await;
        read.extend(print_pair(case, &held, &free));
        check_held(case, &held);
        assert_eq!(free.counts.tool_requests, 0.0, "{}: the free twin holds no tool: {:?}", case.label, free.counts);
        for served in [&held, &free] {
            misses += served.counts.misses;
            repairs += served.counts.repairs;
        }
        drafted += held.counts.drafted;
        accepted += held.counts.accepted;
        for key in ["all".to_owned(), format!("{:?}", case.api)] {
            held_pool.entry(key.clone()).or_default().add(&held.counts);
            free_pool.entry(key).or_default().add(&free.counts);
        }
        if !case.bait.is_empty() {
            let bait = baits.entry(case.bait).or_default();
            bait.0 += 1;
            bait.1 += held.counts.misses;
        }
    }

    eprintln!("\n══ the cost on held traffic (held and free do different work past the first miss: a reading, not a gate)");
    for (key, held) in &held_pool {
        let free = &free_pool[key];
        eprintln!("  {}", held.line(&format!("{key}, held")));
        eprintln!("  {}", free.line(&format!("{key}, free")));
        eprintln!("  {key}: held / free decode rate {:.4}", if free.rate() > 0.0 { held.rate() / free.rate() } else { 0.0 });
    }
    eprintln!("\n══ the misses of the cases built to miss");
    for (bait, (cases, misses)) in &baits {
        eprintln!("  {bait}: {cases} cases, {misses} misses");
        if *misses == 0.0 {
            eprintln!("  NO MISS: the {bait} cases drew no miss; no repair of that kind is read here");
        }
    }
    eprintln!("\n══ the early closes: a required parameter the free twin left out, which the held call wrote");
    let missing = || read.iter().filter(|r| r.fault.kind == "missing");
    let early: Vec<&Read> = missing().filter(|r| r.held != "(absent)" && r.held != "(no such call)").collect();
    for r in &early {
        eprintln!("  {}: {} -> held wrote {}", r.case, r.fault.at, r.held);
    }
    eprintln!("  {} early closes repaired into writing the parameter, of {} parameters the free twins left out", early.len(), missing().count());
    eprintln!("\n══ the other repairs, as the free twins show them");
    for r in read.iter().filter(|r| r.fault.kind != "missing") {
        let bait = if r.bait.is_empty() { String::new() } else { format!(" [{}]", r.bait) };
        eprintln!("  {}{bait}: {} at {} ({}): free wrote {}, held wrote {}", r.case, r.fault.kind, r.fault.at, r.fault.keyword, r.free, r.held);
    }
    eprintln!("\n{done} requests, {misses} misses, {repairs} repairs; the held requests drafted {drafted}, accepted {accepted}");
    assert_eq!(repairs, misses, "every miss repaired");
    assert!(drafted > 0.0 && accepted > 0.0, "the held requests' verify rounds kept their drafts: {drafted} drafted, {accepted} accepted");
}

/// `parallel_tool_calls: false` returns one call per turn on the served
/// 27B: on chat completions and on the Responses API, its tools held
/// (`strict: true`) and free (`strict: false`), each conversation beside its
/// control.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn with_parallel_tool_calls_false_the_27b_makes_one_call_per_turn() {
    let Some(harness) = harness() else { return };
    let server = harness.server.as_ref().unwrap();
    let post = |path: &'static str, body: String| send(server, path, body);
    for api in [Api::Chat, Api::Responses] {
        for held in [true, false] {
            let tools: Vec<String> = corpus::PARALLEL_TOOLS
                .iter()
                .map(|tool| {
                    let tool = if api == Api::Chat { (*tool).to_owned() } else { corpus::flat(tool) };
                    if held {
                        tool
                    } else {
                        corpus::strict_false(&tool)
                    }
                })
                .collect();
            let control = corpus::conversation(&post, MODEL, api, &tools, corpus::PARALLEL_PROMPT, None).await;
            let limited = corpus::conversation(&post, MODEL, api, &tools, corpus::PARALLEL_PROMPT, Some(false)).await;
            let what = format!("27B, {api:?}, {}", if held { "strict tools, held" } else { "strict: false, free" });
            corpus::one_call_per_turn(&what, api, &control, &limited);
        }
    }
}
