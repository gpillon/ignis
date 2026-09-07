//! The G2 measurement instrument end to end (P2-05, GitHub #87): a TTFT
//! cell measured through the *real* HTTP transport (`HttpEndpoint` /
//! reqwest) against the in-process mock engine, then two records turned
//! into a G2 verdict.
//!
//! CPU-only, in-process (ADR 0006: no GPU, no model, no external engine).
//! What the mock gives that a unit test cannot is the wire contract the
//! cold-prefix rule rests on: `stream_options.include_usage` going out, and
//! the trailing usage chunk — `prompt_tokens`, and OpenAI's
//! `prompt_tokens_details.cached_tokens` when the engine served part of the
//! prompt from a prefix cache — coming back.

mod common;

use common::MockEngine;
use ignis_bench::client::{Endpoint, HttpEndpoint, Request};
use ignis_bench::g2;
use ignis_bench::trace::RequestClass;
use ignis_bench::ttft::{self, CellSpec, PromptTemplate, Record, TtftConfig};

/// A template whose post-template token count is the content's whitespace
/// word count — exactly how the mock engine counts a prompt's tokens, so a
/// cell that claims N tokens is a cell the mock computes N tokens for.
struct WordTemplate;

impl PromptTemplate for WordTemplate {
    fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
        Ok(content
            .split_whitespace()
            .map(|w| w.bytes().fold(7u32, |a, b| a.wrapping_mul(131).wrapping_add(b as u32)))
            .collect())
    }
}

fn config(session: &str, label: &str, cells: &[u32], samples: usize) -> TtftConfig {
    TtftConfig {
        cells: cells
            .iter()
            .map(|&prompt_tokens| CellSpec { prompt_tokens, samples })
            .collect(),
        max_tokens: 4,
        label: label.into(),
        profile: format!("{label}-test-profile"),
        artifact: "mock.ninfer".into(),
        session: session.into(),
    }
}

fn measure(engine: &MockEngine, cfg: &TtftConfig) -> Record {
    let ep = HttpEndpoint::new(engine.url());
    ttft::measure(
        &ep,
        &WordTemplate,
        engine.state.model().to_string(),
        engine.url().to_string(),
        cfg,
    )
}

#[test]
fn a_cell_measures_cold_samples_of_the_exact_claimed_length() {
    let engine = MockEngine::start();
    let record = measure(&engine, &config("S-cold", "ignis", &[48, 64], 3));

    assert_eq!(record.cells.len(), 2);
    for (cell, expected) in record.cells.iter().zip([48u32, 64]) {
        assert_eq!(cell.prompt_tokens, expected);
        assert_eq!(cell.samples.len(), 3, "three samples after the warmup");
        assert!(cell.warmup_ttft_ms.is_some(), "the warmup is recorded, never counted");
        assert!(
            cell.all_cold(),
            "every sample must be cold: {:?}",
            cell.void_samples()
        );
        for sample in &cell.samples {
            assert_eq!(
                sample.computed_prefill_tokens,
                Some(expected),
                "the engine must report computing the whole prompt"
            );
        }
        // The statistic is the median of the samples, not of the warmup.
        let ttfts: Vec<f64> = cell.samples.iter().map(|s| s.ttft_ms).collect();
        assert_eq!(cell.median_ttft_ms, ttft::median(&ttfts));
    }
    // Identity: everything a later audit needs to know what was measured.
    assert_eq!(record.session, "S-cold");
    assert_eq!(record.label, "ignis");
    assert_eq!(record.engine, engine.state.model());
    assert_eq!(record.artifact, "mock.ninfer");
    assert_eq!(record.profile, "ignis-test-profile");
    assert!(record.date.ends_with('Z'), "an RFC 3339 UTC date: {}", record.date);
}

#[test]
fn every_sample_including_the_warmup_reached_the_engine_with_its_own_prompt() {
    let engine = MockEngine::start();
    let record = measure(&engine, &config("S-distinct", "ignis", &[40], 5));
    assert!(record.cells[0].all_cold());

    // Six prompts on the wire: the warmup plus five samples, each one a
    // distinct string diverging at its very first word.
    let prompts = engine.state.prompts();
    assert_eq!(prompts.len(), 6, "one warmup + five samples");
    for i in 0..prompts.len() {
        for j in (i + 1)..prompts.len() {
            assert_ne!(prompts[i], prompts[j], "prompts {i} and {j} must differ");
            let first = |p: &String| p.split_whitespace().next().unwrap_or("").to_string();
            assert_ne!(
                first(&prompts[i]),
                first(&prompts[j]),
                "prompts {i} and {j} must diverge at the first content token"
            );
        }
    }
}

#[test]
fn a_sample_served_from_a_prefix_cache_is_void_and_fails_its_cell() {
    let engine = MockEngine::start();
    // The engine reports that it computed only 8 of the prompt's tokens —
    // the rest came from its prefix cache. That is not a measurement of
    // prefill, so the samples are void and the cell is unusable.
    engine.state.set_cached_prompt_tokens(Some(32));
    let record = measure(&engine, &config("S-hot", "ignis", &[40], 3));

    let cell = &record.cells[0];
    assert_eq!(cell.void_samples().len(), 3, "every sample is void");
    assert!(!cell.all_cold());
    for sample in &cell.samples {
        assert_eq!(sample.computed_prefill_tokens, Some(8));
        let reason = sample.void_reason.as_deref().expect("a reason");
        assert!(reason.contains("cache"), "{reason}");
    }
}

#[test]
fn an_engine_that_reports_no_usage_cannot_prove_a_cold_prefix() {
    let engine = MockEngine::start();
    let ep = HttpEndpoint::new(engine.url());
    // A cell measured without the trailing usage chunk has no evidence to
    // read back. `ttft::measure_cell` always asks for it, so the way to
    // reach this state is an engine that ignores `stream_options` — proved
    // here through the outcome shape the instrument reads.
    let outcome = ep
        .complete(&Request {
            id: "no-usage".into(),
            class: RequestClass::Main,
            prompt: "one two three".into(),
            max_tokens: 2,
            stream: true,
            include_usage: false,
            enable_thinking: Some(false),
        })
        .expect("the completion");
    assert_eq!(
        outcome.computed_prefill_tokens(),
        None,
        "no usage chunk means no cold-prefix evidence"
    );
}

#[test]
fn two_live_records_produce_a_g2_verdict() {
    let ours_engine = MockEngine::start();
    let reference_engine = MockEngine::start();
    let session = "S-live";
    let ours = measure(&ours_engine, &config(session, "ignis", &[40, 64], 3));
    let reference = measure(&reference_engine, &config(session, "reference", &[40, 64], 3));

    let verdict = g2::check(&ours, &reference).expect("a verdict");
    assert_eq!(verdict.session, session);
    assert_eq!(verdict.cells.len(), 2);
    for cell in &verdict.cells {
        assert!(cell.ratio > 0.0, "a real ratio: {}", cell.ratio);
    }
    // Both sides are the same mock engine, so the ratio is around 1 and the
    // gate passes — the point here is the pipeline, not the number.
    assert!(verdict.passed, "verdict:\n{}", verdict.render());
}

#[test]
fn a_contaminated_record_is_refused_a_verdict_rather_than_failing_one() {
    let ours_engine = MockEngine::start();
    let reference_engine = MockEngine::start();
    let session = "S-contaminated";
    let ours = measure(&ours_engine, &config(session, "ignis", &[40], 3));
    // The reference served part of every prompt from its prefix cache.
    reference_engine.state.set_cached_prompt_tokens(Some(16));
    let reference = measure(&reference_engine, &config(session, "reference", &[40], 3));

    let refusal = g2::check(&ours, &reference).expect_err("must refuse");
    assert!(refusal.0.contains("void sample"), "{refusal}");
}

#[test]
fn a_record_round_trips_through_json() {
    let engine = MockEngine::start();
    let record = measure(&engine, &config("S-json", "ignis", &[40], 2));
    let json = record.to_json().expect("serialize");
    // Exact: a record is an audit artifact, so what is read back is what was
    // measured, down to the last bit of every TTFT. That holds because the
    // crate builds serde_json with `float_roundtrip` (see its Cargo.toml) --
    // the default parser is allowed to land one ULP off, which would make
    // this assertion flaky.
    assert_eq!(Record::from_json(&json).expect("parse"), record);
}
