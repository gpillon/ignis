//! ignis-bench CLI: trace-replay harness + canary suite + performance report.
//!
//! Subcommands:
//!   `replay  --trace <trace.jsonl> --endpoint <url> [--conc N] [--scale F]
//!             [--out <run.json>] [--label L]`
//!       Re-send the JSONL trace against a running endpoint and write the run
//!       (per-request metrics) as JSON.
//!   `canary  --endpoint <url>`
//!       Run the canary suite (self-consistency: sane + greedy-deterministic)
//!       against a running endpoint.
//!   `report  --ours <ours.json> --ref <ref.json>`
//!       Compare two runs and print the performance report + the 99% gate
//!       verdict (ADR 0007). Exits non-zero when the gate fails.
//!
//! `replay`/`canary` drive a live endpoint through the real `HttpEndpoint`
//! (the `ignis-server`'s OpenAI-compatible API); `report` is fully
//! functional on its own (it just compares two recorded runs).

use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use ignis_bench::{
    canary,
    client::{replay, Endpoint, HttpEndpoint, ReplayConfig},
    metrics::Run,
    report::PerformanceReport,
    trace::Trace,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("replay") => cmd_replay(&args[1..]),
        Some("canary") => cmd_canary(&args[1..]),
        Some("report") => cmd_report(&args[1..]),
        _ => {
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage:
  ignis-bench replay --trace <trace.jsonl> --endpoint <url> [--conc N] [--scale F] [--out <run.json>] [--label L]
  ignis-bench canary --endpoint <url>
  ignis-bench report --ours <ours.json> --ref <ref.json>"
    );
}

/// Return the value following `key` in an argument list, or `None`.
fn opt(args: &[String], key: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == key {
            return args.get(i + 1).cloned();
        }
        i += 1;
    }
    None
}

fn require(args: &[String], key: &str) -> Result<String, String> {
    opt(args, key).ok_or_else(|| format!("--{key} is required"))
}

fn cmd_replay(args: &[String]) -> ExitCode {
    let trace_path = match require(args, "trace") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let endpoint = opt(args, "endpoint").unwrap_or_else(|| "http://127.0.0.1:8080".into());
    let conc = opt(args, "conc").and_then(|v| v.parse::<usize>().ok()).unwrap_or(8);
    let scale = opt(args, "scale").and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
    let label = opt(args, "label").unwrap_or_else(|| "ignis".into());
    let out = opt(args, "out");

    let trace = match Trace::from_path(Path::new(&trace_path)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let ep = HttpEndpoint::new(&endpoint);
    // Pre-flight: the engine must be reachable and have a model loaded — a
    // clean error beats N failed requests in the run file.
    match ep.list_models() {
        Ok(models) => eprintln!("engine {endpoint}: {}", models.join(", ")),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    }
    let ep: Arc<dyn Endpoint> = Arc::new(ep);
    let cfg = ReplayConfig {
        max_concurrency: conc,
        time_scale: scale,
    };
    let metrics = replay(ep, &trace, &cfg);
    let run = Run::new(label, metrics);
    let json = serde_json::to_string_pretty(&run).expect("serialize run");

    match out {
        Some(path) => match std::fs::write(&path, &json) {
            Ok(()) => eprintln!("wrote {path}"),
            Err(e) => {
                eprintln!("error: write {path}: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => println!("{json}"),
    }
    ExitCode::SUCCESS
}

fn cmd_canary(args: &[String]) -> ExitCode {
    let endpoint = match require(args, "endpoint") {
        Ok(u) => u,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let ep = HttpEndpoint::new(&endpoint);
    // Pre-flight: the engine must be reachable and have a model loaded — a
    // clean error beats a full canary run that fails on every request.
    match ep.list_models() {
        Ok(models) => eprintln!("engine {endpoint}: {}", models.join(", ")),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    }
    let ep: Arc<dyn Endpoint> = Arc::new(ep);
    let results = canary::run_canaries(&*ep);
    let consistent = canary::suite_consistent(&results);
    for r in &results {
        println!(
            "canary {:<12} sane={} deterministic={}{}",
            r.id,
            r.sane,
            r.deterministic,
            if r.consistent() { "" } else {"  <-- DIVERGENT"}
        );
    }
    println!("self-consistency: {}", if consistent { "PASS" } else { "FAIL" });
    if consistent {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn cmd_report(args: &[String]) -> ExitCode {
    let ours_path = match require(args, "ours") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let ref_path = match require(args, "ref") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let ours = match load_run(&ours_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let reference = match load_run(&ref_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let report = PerformanceReport::new(&ours, &reference);
    print!("{}", report.render());
    if report.gate_passed() {
        ExitCode::SUCCESS
    } else {
        eprintln!("gate FAILED (ADR 0007: >= {}% of reference speed)", report.threshold * 100.0);
        ExitCode::FAILURE
    }
}

fn load_run(path: &str) -> Result<Run, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))
}