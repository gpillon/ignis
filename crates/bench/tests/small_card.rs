//! The `small-card` preset end to end (#340, spec 27b-16gb/05 Part B AC 7):
//! the whole record assembled over the real HTTP transport against the
//! in-process mock engine, extended with the server's `/v1/tokenize`,
//! `/v1/models/catalog`, `/v1/config`, `/v1/config/origins` and `/metrics`.
//! CPU-only (ADR 0006).
//!
//! What a unit test cannot give: that the pieces are read from the routes the
//! server really serves, that the agent turns go out as recorded chat bodies
//! (tools and all), and that a missing piece degrades to `null` + a note.

mod common;

use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, OnceLock};

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use common::{MockEngine, MockState, ADAPTED, DECODE_LANES, MAX_MODEL_LEN};
use ignis_bench::small_card::{self, MetricsAt, Options};
use serde_json::{json, Value};

/// `/v1/tokenize` as the mock counts a prompt: one token per word, the ids
/// hashed from the words (the server's `return_token_ids` shape).
async fn tokenize(Json(body): Json<Value>) -> Json<Value> {
    let text = body["messages"][0]["content"].as_str().unwrap_or("").to_string();
    let ids: Vec<u32> =
        text.split_whitespace().map(|w| w.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32))).collect();
    Json(json!({ "count": ids.len(), "max_model_len": MAX_MODEL_LEN, "token_ids": ids }))
}

/// The config, its metrics listener on the mock's own port (which serves
/// `/metrics` too): the port is known once the mock listens.
async fn config(State(port): State<Arc<AtomicU16>>) -> Json<Value> {
    let metrics_bind = format!("0.0.0.0:{}", port.load(Ordering::Relaxed));
    Json(json!({
        "server": { "bind": "0.0.0.0:8080", "api_key": "sk-must-not-leak", "metrics": true, "metrics_bind": metrics_bind },
        "model": {
            "artifact": "C:/Users/someone/x.ninfer", "max_context": 262144, "prefill_chunk": 1024,
            // the catalog says the loaded model is Flash-Next's, whose
            // section the record must read (its id says nothing)
            "qwen38flashnext": { "prefill_chunk": 2048 },
        },
        "vram": { "budget_bytes": 17179869184u64 },
        "spec": { "decode_lanes": 2, "draft_head": "auto" },
    }))
}

async fn catalog() -> Json<Value> {
    Json(json!({ "object": "list", "data": [
        { "id": "qwen3.8-27b", "source": "built-in", "on_disk": "absent", "loaded": false },
        { "id": common::MODEL, "source": "known", "on_disk": "complete", "family": "qwen38flashnext", "loaded": true },
    ]}))
}

async fn origins() -> Json<Value> {
    Json(json!({ "object": "list", "data": [
        { "key": "model.max_context", "scope": null, "source": "profile", "scoped": false, "spelling": "rtx-16gb" },
        { "key": "vram.budget_bytes", "scope": null, "source": "flag", "scoped": false, "spelling": "--vram-budget-bytes" },
    ]}))
}

/// The speculative counters stand at 10 rounds, 80 drafted and 60 accepted
/// tokens for each completion request the engine has received (recorded on
/// arrival, so a request the bench has finished is always counted): a decode
/// set's change counts its own requests.
async fn metrics(State(engine): State<Arc<OnceLock<MockState>>>) -> String {
    let n = engine.get().map_or(0, |state| state.bodies().len());
    format!(
        "ignis_build_info{{version=\"0.7.3-test\"}} 1\n\
         ignis_vram_reserved_bytes{{line=\"weights\"}} 12000000000\n\
         ignis_vram_reserved_bytes{{line=\"trellis_workspace\"}} 805306368\n\
         ignis_vram_budget_bytes 17179869184\n\
         ignis_kv_pool_pages 512\n\
         ignis_kv_page_bytes 4194304\n\
         ignis_speculative_rounds_total {}\n\
         ignis_speculative_drafted_tokens_total {}\n\
         ignis_speculative_accepted_tokens_total {}\n",
        10 * n,
        80 * n,
        60 * n
    )
}

/// The server's routes beside the engine's, the cell the config's metrics
/// port is set in once the mock listens, and the one the engine's state is
/// set in for the metrics to count its requests.
fn server_routes() -> (Router, Arc<AtomicU16>, Arc<OnceLock<MockState>>) {
    let port = Arc::new(AtomicU16::new(0));
    let engine = Arc::new(OnceLock::new());
    let routes = Router::new()
        .route("/v1/tokenize", post(tokenize))
        .route("/v1/models/catalog", get(catalog))
        .route("/v1/config", get(config).with_state(Arc::clone(&port)))
        .route("/v1/config/origins", get(origins))
        .route("/metrics", get(metrics).with_state(Arc::clone(&engine)));
    (routes, port, engine)
}

const SMI: &str = "NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 4, 16, 5, 16\n";

fn opts(url: &str, metrics: MetricsAt) -> Options {
    Options { endpoint: url.to_string(), metrics, cells: vec![200, 30_000], samples: 2, decode_tokens: 16 }
}

#[test]
fn the_record_is_assembled_from_what_the_server_exposes() {
    let (routes, port, state) = server_routes();
    let engine = MockEngine::start_with(routes);
    port.store(engine.url().rsplit(':').next().unwrap().parse().unwrap(), Ordering::Relaxed);
    state.set(engine.state.clone()).unwrap();
    // userinfo in the endpoint rides to the server and never into the record;
    // the metrics listener is found where the server's config says
    let url = engine.url().replace("http://", "http://user:secret@");
    let report = small_card::run(&opts(&url, MetricsAt::Server), &|| Ok(SMI.to_string()));
    let json: Value = serde_json::to_value(&report).unwrap();

    // A sample of the record for the report: SMALL_CARD_SAMPLE_OUT=<path>.
    if let Ok(path) = std::env::var("SMALL_CARD_SAMPLE_OUT") {
        std::fs::write(path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }

    assert_eq!(json["schema"], "ignis-small-card/1");
    assert_eq!(json["bench_version"], env!("CARGO_PKG_VERSION"));
    assert!(json["started_at"].as_str().unwrap().ends_with('Z') && json["finished_at"].as_str().unwrap().ends_with('Z'));
    assert!(json["endpoint"].as_str().unwrap().starts_with("http://127.0.0.1:"));

    assert_eq!(json["card"]["name"], "NVIDIA GeForce RTX 5070 Ti");
    assert_eq!(json["card"]["vram_total_mib"], 16303);
    assert_eq!((&json["card"]["pcie_gen_current"], &json["card"]["pcie_width_current"]), (&json!(4), &json!(16)));
    assert_eq!((&json["card"]["pcie_gen_max"], &json["card"]["pcie_width_max"]), (&json!(5), &json!(16)));

    assert_eq!(json["server"]["version"], "0.7.3-test");
    assert_eq!(json["server"]["model_id"], common::MODEL);
    assert_eq!(json["server"]["profile"], "rtx-16gb");
    assert_eq!(json["server"]["max_model_len"], MAX_MODEL_LEN);
    assert_eq!(json["server"]["family"], "qwen38flashnext", "the catalog's loaded row, not the id");
    assert_eq!(json["plan"]["budget_bytes"], 17_179_869_184u64);
    assert_eq!(json["plan"]["reserved_bytes_by_line"]["trellis_workspace"], 805_306_368u64);
    assert_eq!(json["plan"]["kv_pool_pages"], 512);
    assert_eq!(json["config"]["spec.decode_lanes"]["value"], 2);
    assert_eq!(json["config"]["model.prefill_chunk"]["value"], 2048, "the family's section");
    assert_eq!(json["config"]["vram.budget_bytes"]["spelling"], "--vram-budget-bytes");
    // what the load runs at and what it adapted, as /v1/models says
    assert_eq!(json["adaptation"]["configured_max_context"], 262144);
    assert_eq!(json["adaptation"]["running_max_model_len"], MAX_MODEL_LEN);
    assert_eq!(json["adaptation"]["configured_decode_lanes"], 2);
    assert_eq!(json["adaptation"]["running_decode_lanes"], DECODE_LANES);
    assert_eq!(json["adaptation"]["adapted"], ADAPTED);
    assert_eq!(json["adaptation"]["lowered"], true);
    let notes: Vec<&str> = json["notes"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(notes.iter().all(|n| !n.starts_with("adaptation") && !n.starts_with("family")), "{notes:?}");

    // decode: the four coding prompts, the three agent turns and the four
    // long replies, all timed, in that order in the record
    let record = small_card::to_json(&report).unwrap();
    let at: Vec<usize> = ["\"coding\": {", "\"agent\": {", "\"long\": {"]
        .iter()
        .map(|key| record.find(key).unwrap_or_else(|| panic!("no {key} in the record")))
        .collect();
    assert!(at.windows(2).all(|w| w[0] < w[1]), "decode's cells out of order: {at:?}");
    for (set, n) in [("coding", 4), ("agent", 3), ("long", 4)] {
        let samples = json["decode"][set]["samples"].as_array().unwrap();
        assert_eq!(samples.len(), n, "{set}");
        assert!(samples.iter().all(|s| s["tokens_per_s"].as_f64().is_some_and(|r| r > 0.0)), "{set}: {samples:?}");
        assert!(json["decode"][set]["median_tokens_per_s"].as_f64().unwrap() > 0.0, "{set}");
        assert!(json["decode"][set]["pooled_tokens_per_s"].as_f64().unwrap() > 0.0, "{set}");
        assert_eq!(json["decode"][set]["errors"], json!([]), "{set}");
    }
    // each set carries the change across its own requests (the mock's
    // counters count requests): coding 4, agent 3, long 4; the total, across
    // the three sets, is their sum, the warmup's request before them left out
    for (set, requests) in [("coding", 4), ("agent", 3), ("long", 4)] {
        let spec = &json["decode"][set]["speculative"];
        let want = (json!(10 * requests), json!(80 * requests), json!(60 * requests));
        assert_eq!((&spec["rounds"], &spec["drafted_tokens"], &spec["accepted_tokens"]), (&want.0, &want.1, &want.2), "{set}");
        assert_eq!(spec["acceptance"], 0.75, "{set}");
    }
    // the mock stops every reply at 32 tokens: each long reply ends under 256,
    // stays in its cell and leaves a note naming its prompt; the coding and
    // agent prompts, shorter by design, leave none
    let short: Vec<&str> = notes.iter().copied().filter(|n| n.starts_with("decode long:")).collect();
    assert_eq!(short.len(), 4, "{notes:?}");
    for (sample, note) in json["decode"]["long"]["samples"].as_array().unwrap().iter().zip(&short) {
        assert_eq!(sample["completion_tokens"], 32, "{sample}");
        let prompt = sample["prompt"].as_str().unwrap();
        assert!(note.contains(prompt) && note.contains("32 tokens") && note.contains("under 256"), "{note}");
    }
    for set in ["coding", "agent"] {
        for sample in json["decode"][set]["samples"].as_array().unwrap() {
            let prompt = sample["prompt"].as_str().unwrap();
            assert!(notes.iter().all(|n| !n.contains(prompt)), "{prompt} is not a long reply: {notes:?}");
        }
    }
    let total = &json["decode"]["speculative"];
    assert_eq!((&total["rounds"], &total["drafted_tokens"], &total["accepted_tokens"]), (&json!(110), &json!(880), &json!(660)));
    assert_eq!(total["acceptance"], 0.75);
    assert_eq!(total["accepted_per_round"], 6.0);

    // ttft: the small cell is measured cold, the large one does not fit and
    // says which cell would
    let cells = json["ttft"].as_array().unwrap();
    assert_eq!(cells[0]["prompt_tokens"], 200);
    assert!(cells[0]["median_ttft_ms"].as_f64().unwrap() > 0.0);
    assert_eq!(cells[0]["all_cold"], true);
    assert_eq!(cells[0]["void_samples"], 0);
    assert_eq!(cells[0]["samples_ms"].as_array().unwrap().len(), 2);
    assert_eq!(cells[1]["prompt_tokens"], 30_000);
    assert_eq!(cells[1]["median_ttft_ms"], Value::Null);
    let why = cells[1]["error"].as_str().unwrap();
    assert!(why.contains("does not fit") && why.contains("--cells 19992"), "{why}");

    // the decode requests in the order sent: the warmup, the coding prompts,
    // the agent turns, then the long replies, with the TTFT cells after them
    let bodies = engine.state.bodies();
    let canaries: Vec<&str> = ignis_bench::canary::CANARIES.iter().map(|c| c.prompt).collect();
    let (agent_at, long_at) = (1 + canaries.len(), 1 + canaries.len() + 3);
    let (warmup, coding, agent, long) = (&bodies[0], &bodies[1..agent_at], &bodies[agent_at..long_at], &bodies[long_at..long_at + 4]);
    assert_eq!((&warmup["max_tokens"], &warmup["enable_thinking"]), (&json!(16), &json!(false)));
    // the coding prompts: the canaries, the --decode-tokens cap, thinking
    // off, streaming
    let sent: Vec<&str> = coding.iter().map(|b| b["messages"][0]["content"].as_str().unwrap()).collect();
    assert_eq!(sent, canaries);
    assert!(coding.iter().all(|b| b["max_tokens"] == 16 && b["stream"] == true && b["enable_thinking"] == false));
    // the agent turns went out as recorded bodies: tools and history intact
    assert!(agent.iter().all(|b| b.get("tools").is_some()), "three agent turns sent with their tools");
    assert!(agent.iter().all(|b| b["messages"].as_array().unwrap().iter().any(|m| m["role"] == "tool")));
    assert!(agent.iter().all(|b| b["max_tokens"] == 16 && b["stream"] == true));
    // the agent turns think, so their reasoning streams (a tool call comes
    // whole); every other request has thinking off
    assert!(agent.iter().all(|b| b["enable_thinking"] == true), "{agent:?}");
    // the long replies: one user message each, the preset's own cap whatever
    // --decode-tokens says, thinking off, streaming, and nothing that makes
    // the server generate past the reply's end
    for b in long {
        assert_eq!(b["messages"].as_array().unwrap().len(), 1, "{b}");
        assert_eq!(b["messages"][0]["role"], "user", "{b}");
        assert_eq!((&b["max_tokens"], &b["stream"], &b["enable_thinking"]), (&json!(1024), &json!(true), &json!(false)), "{b}");
        assert!(b.get("ignore_eos").is_none() && b.get("tools").is_none(), "{b}");
    }
    let others: Vec<&Value> = bodies.iter().filter(|b| b.get("tools").is_none()).collect();
    assert!(others.iter().all(|b| b["enable_thinking"] == false), "{others:?}");
    assert!(bodies.iter().all(|b| b.get("ignore_eos").is_none()), "no request asks for ignore_eos");

    // nothing private reaches the record
    let text = serde_json::to_string(&json).unwrap();
    for leak in ["secret", "sk-must-not-leak", "someone", "api_key", "authorization", "Authorization", "0.0.0.0"] {
        assert!(!text.contains(leak), "{leak} leaked into the record");
    }
}

#[test]
fn a_missing_piece_is_null_with_a_note_and_the_rest_is_still_measured() {
    // The plain mock: models and completions only. No config, no tokenizer,
    // no metrics listener; and no nvidia-smi.
    let engine = MockEngine::start();
    let report = small_card::run(&opts(engine.url(), MetricsAt::Off), &|| Err("could not run nvidia-smi: not found".to_string()));
    let json: Value = serde_json::to_value(&report).unwrap();

    assert_eq!(json["card"], Value::Null);
    assert_eq!(json["config"], Value::Null);
    assert_eq!(json["server"]["version"], Value::Null);
    assert_eq!(json["server"]["profile"], Value::Null);
    assert_eq!(json["plan"]["budget_bytes"], Value::Null);
    assert_eq!(json["plan"]["reserved_bytes_by_line"], Value::Null);
    assert_eq!(json["decode"]["speculative"], Value::Null);
    for set in ["coding", "agent", "long"] {
        let cell = json["decode"][set].as_object().unwrap();
        assert_eq!(cell.get("speculative"), Some(&Value::Null), "{set}: the key is there, null");
    }
    assert_eq!(json["adaptation"]["configured_max_context"], Value::Null);
    // /v1/models still says what the load runs at
    assert_eq!(json["adaptation"]["running_decode_lanes"], DECODE_LANES);
    assert_eq!(json["adaptation"]["lowered"], true);
    // the same keys are there, null, not absent
    assert!(json.as_object().unwrap().contains_key("card") && json["plan"].as_object().unwrap().contains_key("kv_pool_pages"));

    let notes: Vec<&str> = json["notes"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    for want in ["card:", "config:", "metrics: not read", "tokenize", "family: /v1/models/catalog could not say"] {
        assert!(notes.iter().any(|n| n.contains(want)), "no note for `{want}` in {notes:?}");
    }
    // the cells say why they have no number
    let cells = json["ttft"].as_array().unwrap();
    assert!(cells[0]["error"].as_str().unwrap().contains("tokenizer"), "{:?}", cells[0]);
    assert_eq!(cells[0]["median_ttft_ms"], Value::Null);
    // decode needs only the completions route
    assert_eq!(json["decode"]["coding"]["samples"].as_array().unwrap().len(), 4);
    assert!(json["decode"]["agent"]["median_tokens_per_s"].as_f64().unwrap() > 0.0);
}

#[test]
fn an_unreachable_server_yields_a_record_that_says_so_and_does_not_panic() {
    // Port 1 on localhost refuses at once.
    let report = small_card::run(&opts("http://127.0.0.1:1", MetricsAt::Off), &|| Ok(SMI.to_string()));
    assert_eq!(report.server.model_id, None);
    assert!(report.decode.coding.is_none() && report.ttft.is_empty());
    assert!(report.notes.iter().any(|n| n.contains("nothing was measured")), "{:?}", report.notes);
    assert_eq!(report.card.as_ref().and_then(|c| c.vram_total_mib), Some(16303));
}
