//! The canary oracle (`ignis-bench oracle`, P1-04, GitHub #40) against an
//! in-process mock engine: records a fixture through the real `HttpEndpoint`
//! transport, then compares it against a second recording of the *same*
//! mock (should agree fully) and against a hand-edited candidate (should
//! diverge) — end to end, CPU-only, no GPU and no `.ninfer` artifact (ADR
//! 0006; prior art: `record_capture.rs`'s mock-target test).

mod common;

use common::MockEngine;
use ignis_bench::canary::CANARIES;
use ignis_bench::client::HttpEndpoint;
use ignis_bench::oracle::{self, Tokenize};

/// A trivial deterministic tokenizer for this CPU test: one id per
/// whitespace word (a stable checksum of the word), so identical text always
/// tokenizes identically and different text (almost always) differs — no
/// `.ninfer` artifact needed for a fully-CPU integration test.
struct WordIdTokenizer;

impl Tokenize for WordIdTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text
            .split_whitespace()
            .map(|w| w.bytes().fold(0u32, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u32)))
            .collect())
    }
}

#[test]
fn record_then_compare_against_the_same_mock_engine_agrees_fully() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url());
    let models = ep.list_models().expect("the mock reports a model");
    let model = models[0].clone();

    // The "reference" recording.
    let oracle_fixture =
        oracle::record(&ep, &WordIdTokenizer, model.clone(), 8).expect("record the oracle fixture");
    assert_eq!(oracle_fixture.prompts.len(), CANARIES.len());
    assert_eq!(oracle_fixture.max_tokens, 8);

    // The fixture round-trips through disk (the on-disk format `oracle
    // record`/`oracle compare` actually exchange).
    let path = std::env::temp_dir().join(format!(
        "ignis-bench-oracle-it-{}.json",
        std::process::id()
    ));
    oracle_fixture.write(&path).expect("write the fixture");
    let reloaded = ignis_bench::oracle::Fixture::read(&path).expect("read the fixture back");
    assert_eq!(reloaded, oracle_fixture);
    let _ = std::fs::remove_file(&path);

    // A second recording against the *same* deterministic mock: the mock's
    // completion text is a pure function of `max_tokens` (see
    // `common::token_text`), so a fresh recording reproduces the same text
    // and therefore the same token ids — full agreement, no divergence.
    let candidate_fixture =
        oracle::record(&ep, &WordIdTokenizer, model, 8).expect("record the candidate fixture");
    let results = oracle::compare_fixtures(&oracle_fixture, &candidate_fixture, 8)
        .expect("both fixtures cover every canary");
    assert_eq!(results.len(), CANARIES.len());
    for r in &results {
        assert!(r.is_full_agreement(), "canary {}: expected full agreement, got {r:?}", r.id);
    }
    assert_eq!(oracle::overall_agreement(&results), 1.0);
}

#[test]
fn a_diverging_candidate_is_caught_with_its_first_divergence_position() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url());
    let model = ep.list_models().expect("the mock reports a model")[0].clone();

    let oracle_fixture =
        oracle::record(&ep, &WordIdTokenizer, model, 8).expect("record the oracle fixture");

    // A candidate that agrees on the first canary but diverges on the rest
    // (simulating a broken engine that only gets the first prompt right).
    let mut candidate_fixture = oracle_fixture.clone();
    for p in candidate_fixture.prompts.iter_mut().skip(1) {
        for id in p.token_ids.iter_mut() {
            *id = id.wrapping_add(1);
        }
    }

    let results = oracle::compare_fixtures(&oracle_fixture, &candidate_fixture, 8)
        .expect("both fixtures cover every canary");
    assert!(results[0].is_full_agreement(), "the first canary was left untouched");
    for r in &results[1..] {
        assert_eq!(r.first_divergence, Some(0), "canary {}: diverges immediately", r.id);
        assert_eq!(r.agree, 0);
    }
    let overall = oracle::overall_agreement(&results);
    assert!(overall < 0.95, "a mostly-broken candidate must fail the G1 floor, got {overall}");
}
