//! Integration test: the `report` workflow end-to-end — two recorded runs are
//! serialized to JSON, deserialized back, and the performance report + 99% gate
//! (ADR 0007) is computed. This mirrors the `ignis-bench report` subcommand and
//! pins the JSON round-trip of `Run` / `RequestMetrics` / `RequestClass`.

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

#[test]
fn runs_round_trip_through_json_and_the_gate_is_computed() {
    // A run that has *both* classes, identical to the reference -> the gate
    // must pass (ratio 1.0 >= 0.99, ttft ratio 1.0 <= 1/0.99).
    let ref_run = Run::new(
        "ref",
        vec![
            m("main", RequestClass::Main, 200.0, 512, 6200.0),
            m("s1", RequestClass::Sub, 100.0, 64, 2000.0),
        ],
    );

    // Round-trip both runs through JSON exactly as the CLI does:
    //   `replay` writes a run as JSON; `report` reads two runs back.
    let ours_json = serde_json::to_string_pretty(&ref_run).expect("serialize ours");
    let ref_json = serde_json::to_string_pretty(&ref_run).expect("serialize ref");
    let ours: Run = serde_json::from_str(&ours_json).expect("parse ours");
    let reference: Run = serde_json::from_str(&ref_json).expect("parse ref");

    // The round-trip preserves the per-class structure (classes survive
    // serde via the snake_case rename).
    assert_eq!(ours.classes(), vec![RequestClass::Main, RequestClass::Sub]);

    let report = PerformanceReport::new(&ours, &reference);
    assert!(report.gate_passed(), "identical runs must pass the 99% gate");
    assert!(report.render().contains("gate: PASS"));
}

#[test]
fn a_slower_run_fails_the_gate() {
    // Reference: sub class ~99.9 tok/s (999 decode tokens over 10.0 s).
    let ref_run = Run::new(
        "ref",
        vec![
            m("main", RequestClass::Main, 200.0, 512, 6200.0),
            m("s1", RequestClass::Sub, 100.0, 1000, 10_100.0),
        ],
    );
    // Ours: same main (fine) but a much slower sub class -> sub gate fails.
    let ours_run = Run::new(
        "ours",
        vec![
            m("main", RequestClass::Main, 200.0, 512, 6200.0),
            // 899 decode tokens over 9.9 s ~= 90.8 tok/s (< 99% of 99.9).
            m("s1", RequestClass::Sub, 100.0, 900, 10_000.0),
        ],
    );

    // Round-trip through JSON as the CLI does.
    let ours_json = serde_json::to_string_pretty(&ours_run).expect("serialize ours");
    let ref_json = serde_json::to_string_pretty(&ref_run).expect("serialize ref");
    let ours: Run = serde_json::from_str(&ours_json).expect("parse ours");
    let reference: Run = serde_json::from_str(&ref_json).expect("parse ref");

    let report = PerformanceReport::new(&ours, &reference);
    assert!(!report.gate_passed(), "the slower sub class must fail the gate");
    let sub = report
        .per_class
        .iter()
        .find(|g| g.class == RequestClass::Sub)
        .expect("sub class gate");
    assert!(!sub.tok_s_ok);
    // The main class still passes (it's identical).
    let main = report
        .per_class
        .iter()
        .find(|g| g.class == RequestClass::Main)
        .expect("main class gate");
    assert!(main.passed());
}