//! `ttft::pool_launches`: one engine's TTFT records from independent
//! launches, pooled into the record `g2` compares.

use std::sync::atomic::{AtomicU32, Ordering};

use ignis_bench::ttft::{pool_launches, Cell, Record, Sample};

fn sample(index: usize, ttft_ms: f64, prompt_tokens: u32) -> Sample {
    Sample { index, ttft_ms, computed_prefill_tokens: Some(prompt_tokens), void: false, void_reason: None }
}

fn cell(prompt_tokens: u32, ttfts: &[f64]) -> Cell {
    Cell {
        prompt_tokens,
        warmup_ttft_ms: Some(1.0),
        samples: ttfts.iter().enumerate().map(|(i, t)| sample(i, *t, prompt_tokens)).collect(),
        median_ttft_ms: ignis_bench::ttft::median(ttfts),
        error: None,
        image: None,
    }
}

/// Each launch is made a second after the last one: two independent
/// launches never share a date.
fn launch(session: &str, cells: Vec<Cell>) -> Record {
    static SECONDS: AtomicU32 = AtomicU32::new(0);
    let second = SECONDS.fetch_add(1, Ordering::Relaxed);
    Record {
        session: session.into(),
        label: "ignis".into(),
        endpoint: "http://127.0.0.1:8000".into(),
        engine: "qwen3.8-27b".into(),
        artifact: "model.ninfer".into(),
        profile: "hq-e8-2b KV".into(),
        date: format!("2026-11-27T10:{:02}:{:02}Z", second / 60 % 60, second % 60),
        max_tokens: 8,
        cells,
    }
}

#[test]
fn two_launches_pool_into_one_record_whose_median_is_over_every_sample() {
    // The second launch sits in a slower band: neither launch's own median
    // (20 and 50) is the pooled one.
    let first = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);
    let second = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);

    let pooled = pool_launches(&[first, second]).expect("two launches of one engine pool");

    let cell = pooled.cell(8192).expect("the cell is kept");
    let ttfts: Vec<f64> = cell.samples.iter().map(|s| s.ttft_ms).collect();
    assert_eq!(ttfts, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], "every sample of every launch");
    let indices: Vec<usize> = cell.samples.iter().map(|s| s.index).collect();
    assert_eq!(indices, [0, 1, 2, 3, 4, 5], "numbered across the launches");
    assert_eq!(cell.median_ttft_ms, Some(35.0));
    assert_eq!(pooled.session, "s1");
    assert!(pooled.profile.contains("pooled over 2 launches"), "{}", pooled.profile);
}

#[test]
fn one_launch_alone_is_refused() {
    let only = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);

    let refusal = pool_launches(&[only]).expect_err("one launch is not a pool");
    assert!(refusal.contains("at least two independent launches"), "{refusal}");
    let refusal = pool_launches(&[]).expect_err("no launch is not a pool");
    assert!(refusal.contains("at least two independent launches"), "{refusal}");
}

#[test]
fn launches_that_are_not_one_engine_in_one_session_are_refused() {
    let first = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);

    let other_session = launch("s2", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    let refusal = pool_launches(&[first.clone(), other_session]).expect_err("two sessions");
    assert!(refusal.contains("session") && refusal.contains("s1") && refusal.contains("s2"), "{refusal}");

    let mut other_engine = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    other_engine.label = "another-engine".into();
    let refusal = pool_launches(&[first.clone(), other_engine]).expect_err("two engines");
    assert!(refusal.contains("label") && refusal.contains("another-engine"), "{refusal}");

    let mut other_model = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    other_model.engine = "another-model".into();
    let refusal = pool_launches(&[first.clone(), other_model]).expect_err("two models");
    assert!(refusal.contains("engine") && refusal.contains("another-model"), "{refusal}");

    let mut other_artifact = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    other_artifact.artifact = "another.ninfer".into();
    let refusal = pool_launches(&[first, other_artifact]).expect_err("two artifacts");
    assert!(refusal.contains("artifact") && refusal.contains("another.ninfer"), "{refusal}");
}

#[test]
fn the_same_launch_given_twice_is_refused() {
    let first = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);
    let second = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);

    let refusal = pool_launches(&[first.clone(), first.clone()]).expect_err("one launch twice");
    assert!(
        refusal.contains("launch 2") && refusal.contains("launch 1") && refusal.contains(&first.date),
        "{refusal}"
    );

    // Not only the first launch: any two of them.
    let refusal = pool_launches(&[first, second.clone(), second]).expect_err("the second launch twice");
    assert!(refusal.contains("launch 3") && refusal.contains("launch 2"), "{refusal}");
}

#[test]
fn a_pooled_record_given_as_a_launch_is_refused() {
    let pooled = pool_launches(&[
        launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]),
        launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]),
    ])
    .expect("two launches pool");
    let third = launch("s1", vec![cell(8192, &[70.0, 80.0, 90.0])]);

    let refusal = pool_launches(&[pooled.clone(), third.clone()]).expect_err("a pooled record first");
    assert!(refusal.contains("launch 1") && refusal.contains("pooled over 2 launches"), "{refusal}");
    let refusal = pool_launches(&[third, pooled]).expect_err("a pooled record second");
    assert!(refusal.contains("launch 2") && refusal.contains("pooled over 2 launches"), "{refusal}");

    // The mark is the one `pool_launches` writes, not the words anywhere.
    let mut first = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);
    let mut second = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    first.profile = "hq-e8-2b KV (pooled over the two cards)".into();
    second.profile = first.profile.clone();
    pool_launches(&[first, second]).expect("a profile that only says pooled pools");
}

#[test]
fn launches_of_one_engine_served_or_asked_differently_are_refused() {
    let first = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);

    let mut other_profile = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    other_profile.profile = "hq-e8-2b KV, 4096 chunk".into();
    let refusal = pool_launches(&[first.clone(), other_profile]).expect_err("two profiles");
    assert!(
        refusal.contains("launch 2") && refusal.contains("profile") && refusal.contains("4096 chunk"),
        "{refusal}"
    );

    let mut other_budget = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    other_budget.max_tokens = 64;
    let refusal = pool_launches(&[first, other_budget]).expect_err("two generation budgets");
    assert!(
        refusal.contains("launch 2") && refusal.contains("max_tokens") && refusal.contains("64"),
        "{refusal}"
    );
}

#[test]
fn launches_that_did_not_measure_the_same_cells_are_refused() {
    let first = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0]), cell(32768, &[70.0, 80.0, 90.0])]);

    let one_cell_short = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    let refusal = pool_launches(&[first.clone(), one_cell_short]).expect_err("a missing cell");
    assert!(refusal.contains("launch 2") && refusal.contains("cells"), "{refusal}");

    let another_length = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0]), cell(65536, &[70.0, 80.0, 90.0])]);
    let refusal = pool_launches(&[first, another_length]).expect_err("another cell");
    assert!(refusal.contains("launch 2") && refusal.contains("cells"), "{refusal}");

    // An image cell is named by its length and its image (GitHub #181).
    let image = |sha256: &str| ignis_bench::ttft::ImageIdentity {
        sha256: sha256.into(),
        width: 1280,
        height: 800,
        question: "What is this?".into(),
    };
    let with_image = |sha256: &str| {
        let mut cell = cell(1027, &[10.0, 20.0, 30.0]);
        cell.image = Some(image(sha256));
        launch("s1", vec![cell])
    };
    let refusal = pool_launches(&[with_image("aa"), with_image("bb")]).expect_err("another image");
    assert!(refusal.contains("launch 2") && refusal.contains("cells"), "{refusal}");
    pool_launches(&[with_image("aa"), with_image("aa")]).expect("the same image cell pools");
}

#[test]
fn a_launch_whose_cell_failed_is_refused() {
    let first = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);
    let mut failed = launch("s1", vec![cell(8192, &[])]);
    failed.cells[0].median_ttft_ms = None;
    failed.cells[0].error = Some("the warmup request failed".into());

    let refusal = pool_launches(&[first, failed]).expect_err("a failed cell has nothing to pool");
    assert!(
        refusal.contains("launch 2") && refusal.contains("8192") && refusal.contains("the warmup request failed"),
        "{refusal}"
    );
}

#[test]
fn a_void_sample_stays_void_in_the_pooled_record_so_the_comparison_refuses_it() {
    let first = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);
    let mut warm = launch("s1", vec![cell(8192, &[40.0, 50.0, 60.0])]);
    warm.cells[0].samples[1].void = true;
    warm.cells[0].samples[1].void_reason = Some("part of it was served from a cache".into());

    let pooled = pool_launches(&[first.clone(), warm]).expect("the launches pool");
    assert!(!pooled.cells[0].all_cold(), "the pooled cell is not all cold");

    let again = launch("s1", vec![cell(8192, &[10.0, 20.0, 30.0])]);
    let mut reference = pool_launches(&[first.clone(), again]).expect("the other side");
    reference.label = "another-engine".into();
    let refusal = ignis_bench::g2::check(&pooled, &reference).expect_err("a void sample decides nothing");
    assert!(refusal.0.contains("void sample (#4)"), "{refusal}");
}
