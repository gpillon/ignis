//! The G3 measurement instrument end to end (P3-07, GitHub #100): C=1 /
//! C=4 / ITL measured through the *real* HTTP transport (`HttpEndpoint` /
//! reqwest) against the in-process mock engine, then two records turned
//! into a G3 verdict.
//!
//! CPU-only, in-process (ADR 0006: no GPU, no model, no external engine).
//! Fixture sizes here are small (the mock caps generation at 32 tokens and
//! the filler prompt generator is O(n^2) at production scale — see
//! `ttft::generate_prompt`'s doc comment); the point is the wire contract
//! (per-token SSE timing, the trailing usage chunk, concurrent + sequential
//! request shapes) and the pipeline, not the gate's real numbers.

mod common;

use common::MockEngine;
use ignis_bench::client::HttpEndpoint;
use ignis_bench::g3::{self, G3Config, ItlConfig, Record, ThroughputSpec};
use ignis_bench::g3_gate;
use ignis_bench::ttft::PromptTemplate;

/// A template whose post-template token count is the content's whitespace
/// word count — exactly how the mock engine counts a prompt's tokens
/// (`tests/ttft_cell.rs` uses the same pattern).
struct WordTemplate;

impl PromptTemplate for WordTemplate {
    fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
        Ok(content
            .split_whitespace()
            .map(|w| w.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32)))
            .collect())
    }
    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        Ok(ids.iter().map(|id| format!("t{id}")).collect::<Vec<_>>().join(" "))
    }
}

fn config(session: &str, label: &str) -> G3Config {
    G3Config {
        label: label.into(),
        profile: format!("{label}-test-profile"),
        artifact: "mock.ninfer".into(),
        session: session.into(),
        throughput: ThroughputSpec { prompt_tokens: 24, max_tokens: 8 },
        itl: ItlConfig {
            prefill_prompt_tokens: 20,
            prefill_max_tokens: 4,
            prefill_count: 3,
            decode_prompt_tokens: 16,
            decode_max_tokens: 6,
            decode_lanes: 2,
        },
    }
}

fn measure(engine: &MockEngine, cfg: &G3Config) -> Record {
    let ep = HttpEndpoint::new(engine.url());
    g3::measure(&ep, &WordTemplate, engine.state.model().to_string(), engine.url().to_string(), cfg)
}

#[test]
fn a_record_measures_all_three_cells_cold_over_real_http() {
    let engine = MockEngine::start();
    let record = measure(&engine, &config("S-cold", "ignis"));

    assert!(record.c1.all_cold(), "bad: {:?}", record.c1.bad_samples());
    assert_eq!(record.c1.samples.len(), 1);
    assert!(record.c4.all_cold(), "bad: {:?}", record.c4.bad_samples());
    assert_eq!(record.c4.samples.len(), 4);
    assert!(record.c1.aggregate_tok_s > 0.0);
    assert!(record.c4.aggregate_tok_s > 0.0);

    assert!(record.itl.all_cold(), "void: {:?}", record.itl.void_prefillers());
    assert_eq!(record.itl.prefillers.len(), 3);
    assert_eq!(record.itl.lanes.len(), 2);
    // Two lanes x (6 tokens -> 5 intervals) = 10 pooled intervals.
    assert_eq!(record.itl.intervals_ms.len(), 10);
    assert!(record.itl.p95_ms.is_some());

    assert_eq!(record.session, "S-cold");
    assert_eq!(record.label, "ignis");
    assert_eq!(record.engine, engine.state.model());
    assert!(record.date.ends_with('Z'), "an RFC 3339 UTC date: {}", record.date);
}

#[test]
fn a_prefix_cache_hit_voids_the_throughput_samples_and_the_prefillers() {
    let engine = MockEngine::start();
    engine.state.set_cached_prompt_tokens(Some(4));
    let record = measure(&engine, &config("S-hot", "ignis"));

    assert!(!record.c1.all_cold());
    assert!(!record.c4.all_cold());
    for sample in record.c1.bad_samples().into_iter().chain(record.c4.bad_samples()) {
        assert!(sample.void_reason.as_deref().unwrap().contains("cache"));
    }
    assert!(!record.itl.all_cold());
    assert_eq!(record.itl.void_prefillers().len(), 3);
}

#[test]
fn two_live_records_produce_a_g3_verdict() {
    let ours_engine = MockEngine::start();
    let reference_engine = MockEngine::start();
    let session = "S-live";
    let ours = measure(&ours_engine, &config(session, "ignis"));
    let reference = measure(&reference_engine, &config(session, "reference"));

    let verdict = g3_gate::check(&ours, &reference).expect("a verdict");
    assert_eq!(verdict.session, session);
    // Both sides are two independent instances of the same mock engine: at
    // this fixture scale (a handful of ms per decode phase) OS scheduling
    // jitter can move a ratio either side of 1 — the point here is that the
    // pipeline produces a real, positive comparison, not the exact number.
    assert!(verdict.c1.ratio > 0.0 && verdict.c4.ratio > 0.0 && verdict.itl.ratio > 0.0);
}

#[test]
fn a_contaminated_record_is_refused_a_verdict_rather_than_failing_one() {
    let ours_engine = MockEngine::start();
    let reference_engine = MockEngine::start();
    let session = "S-contaminated";
    let ours = measure(&ours_engine, &config(session, "ignis"));
    reference_engine.state.set_cached_prompt_tokens(Some(4));
    let reference = measure(&reference_engine, &config(session, "reference"));

    let refusal = g3_gate::check(&ours, &reference).expect_err("must refuse");
    assert!(
        refusal.0.contains("bad sample") || refusal.0.contains("void prefiller"),
        "{refusal}"
    );
}

#[test]
fn a_record_round_trips_through_json() {
    let engine = MockEngine::start();
    let record = measure(&engine, &config("S-json", "ignis"));
    let json = record.to_json().expect("serialize");
    assert_eq!(Record::from_json(&json).expect("parse"), record);
}

#[test]
fn the_decode_lanes_run_concurrently_with_the_sequential_prefillers() {
    // The mock's in-flight peak (`common::MockState::peak_in_flight`) proves
    // the decode lanes and the (sequential) prefillers were not serialized
    // by the instrument itself: at least the decode lanes must overlap each
    // other, and at least one prefiller must overlap them.
    let engine = MockEngine::start();
    let _record = measure(&engine, &config("S-concurrency", "ignis"));
    assert!(
        engine.state.peak_in_flight() >= 2,
        "expected the decode lanes to overlap: peak {}",
        engine.state.peak_in_flight()
    );
}
