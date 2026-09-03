//! The `gate` workflow end to end (spec 02, ADR 0007): two recorded runs
//! (as `replay --out` writes them) + the canary results (as `canary --out`
//! writes them) are read back from JSON and the shipped v1 acceptance
//! artifact — the performance report + the divergence report + the overall
//! verdict — is computed. This mirrors the `ignis-bench gate` subcommand:
//! `gate --ours ours.json --ref ref.json [--canary canary.json] [--out gate.json]`.
//!
//! The JSON round-trips stand in for the files on disk (the same seam as
//! `report_from_json.rs`): the artifact the CLI ships *is* this JSON.

use ignis_bench::canary::{evaluate, CanaryResult};
use ignis_bench::gate::GateReport;
use ignis_bench::metrics::{RequestMetrics, Run};
use ignis_bench::report::PerformanceReport;
use ignis_bench::trace::RequestClass;

fn m(id: &str, class: RequestClass, ttft: f64, n: u32, total: f64) -> RequestMetrics {
    RequestMetrics {
        id: id.into(),
        class,
        ttft_ms: ttft,
        n_tokens: n,
        total_ms: total,
        ok: true,
    }
}

/// The runs both engines produced on the load (both classes present, as
/// `replay --out` records them). Identical on purpose: the 99% performance
/// gate holds (ratio 1.0), so the verdicts exercised here isolate the
/// self-consistency half.
fn runs() -> (Run, Run) {
    (
        Run::new(
            "ignis",
            vec![
                m("main", RequestClass::Main, 200.0, 512, 6200.0),
                m("s1", RequestClass::Sub, 100.0, 64, 2000.0),
            ],
        ),
        Run::new(
            "ninfer",
            vec![
                m("main", RequestClass::Main, 200.0, 512, 6200.0),
                m("s1", RequestClass::Sub, 100.0, 64, 2000.0),
            ],
        ),
    )
}

/// A consistent canary suite (greedy + fixed seed ⇒ identical outputs).
fn consistent_canaries() -> Vec<CanaryResult> {
    vec![
        evaluate("rust-hello", "it prints `hi`", "it prints `hi`"),
        evaluate("math-greedy", "the answer is 10", "the answer is 10"),
    ]
}

/// A divergent canary suite: one canary's two greedy runs disagree (the
/// divergence the suite exists to detect).
fn divergent_canaries() -> Vec<CanaryResult> {
    vec![
        evaluate("rust-hello", "it prints `hi`", "it prints `hi`"),
        evaluate("math-greedy", "the answer is 10", "the answer is 11"),
    ]
}

/// Load `ours` / `reference` / the canary results from JSON, exactly as the
/// `gate` subcommand does (the `replay --out` / `canary --out` files). The
/// v1 verdict is the *conjunction* of the performance gate and the
/// self-consistency check (spec 02), so the canary file is always present.
fn load_shipped(ours: &Run, reference: &Run, canary: &[CanaryResult]) -> GateReport {
    let ours_json = serde_json::to_string_pretty(ours).expect("serialize ours");
    let ref_json = serde_json::to_string_pretty(reference).expect("serialize reference");
    let ours: Run = serde_json::from_str(&ours_json).expect("parse ours");
    let reference: Run = serde_json::from_str(&ref_json).expect("parse reference");
    let canary_json = serde_json::to_string_pretty(canary).expect("serialize canaries");
    let canary_back: Vec<CanaryResult> =
        serde_json::from_str(&canary_json).expect("parse canaries");
    GateReport::new(PerformanceReport::new(&ours, &reference), canary_back)
}

#[test]
fn the_v1_gate_artifact_is_built_from_shipped_json() {
    // The v1 acceptance (spec 02, ADR 0007): the performance gate holds
    // (identical runs) and the canaries are self-consistent -> the shipped
    // artifact's verdict is PASS.
    let (ours, reference) = runs();
    let artifact = load_shipped(&ours, &reference, &consistent_canaries());
    assert!(
        artifact.performance.gate_passed(),
        "the 99% performance gate holds"
    );
    assert!(artifact.passed(), "the v1 gate must pass");

    // The artifact itself is the shipped file: it round-trips through JSON
    // (`ignis-bench gate --out` writes it, the next run reads it back).
    let shipped = serde_json::to_string_pretty(&artifact).expect("ship the artifact");
    let back: GateReport = serde_json::from_str(&shipped).expect("parse the shipped artifact");
    assert!(back.passed(), "the shipped artifact keeps its verdict");
    assert_eq!(
        back.canary.len(),
        2,
        "the divergence report survives the round-trip"
    );
}

#[test]
fn a_divergent_canary_fails_the_shipped_v1_gate() {
    // The performance gate holds (identical runs), but one canary is not
    // deterministic (greedy drift) -> the self-consistency check fails and
    // the overall v1 verdict is FAIL (ADR 0007: correctness is
    // self-checked, not reference-matched).
    let (ours, reference) = runs();
    let artifact = load_shipped(&ours, &reference, &divergent_canaries());
    assert!(
        artifact.performance.gate_passed(),
        "the performance gate still holds"
    );
    assert!(
        !artifact.passed(),
        "a divergent canary must fail the v1 gate"
    );
    let text = artifact.render();
    assert!(
        text.contains("v1 gate: FAIL"),
        "the rendered verdict is FAIL: {text}"
    );
    assert!(
        text.contains("canary (self-consistency)"),
        "the divergence report is part of the shipped artifact"
    );
}