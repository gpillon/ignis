//! ignis-bench CLI: trace-replay harness + canary suite + v1 gate.
//!
//! Subcommands:
//!   `replay  --trace <trace.jsonl> --endpoint <url> [--conc N] [--scale F]
//!             [--out <run.json>] [--label L]`
//!       Re-send the JSONL trace against a running endpoint and write the run
//!       (per-request metrics) as JSON.
//!   `canary  --endpoint <url> [--out <canary.json>]`
//!       Run the canary suite (self-consistency: sane + greedy-deterministic)
//!       against a running endpoint; ship the divergence report as JSON.
//!   `report  --ours <ours.json> --ref <ref.json>`
//!       Compare two runs and print the performance report + the 99% gate
//!       verdict (ADR 0007). Exits non-zero when the gate fails.
//!   `gate    --ours <ours.json> --ref <ref.json> --canary <canary.json>
//!             [--out <gate.json>]`
//!       The v1 acceptance artifact (ADR 0007): the performance report (the
//!       99% gate, per class) + the divergence report (canary
//!       self-consistency — the v1 verdict is their conjunction), shipped
//!       as a single JSON file. Exits non-zero when the v1 verdict fails.
//!   `record  --listen <proxy> --target <engine-url> --out <load>-trace.jsonl
//!             [--class <policy>]`
//!       The capture proxy (spec 03): accepts OpenAI chat-completions from a
//!       live agent client, records each request as a trace line, forwards
//!       it to the target engine, and pipes the response back. `POST
//!       /v1/session/end` finalizes the trace and stops the proxy.
//!
//! `replay`/`canary` drive a live endpoint through the real `HttpEndpoint`
//! (the `ignis-server`'s OpenAI-compatible API); `report`/`gate` are fully
//! functional on their own (they compare recorded runs and canary results);
//! `record` captures a live session (the gate-run's capture piece, spec 03).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use ignis_bench::{
    canary::{self, CanaryResult},
    client::{replay, Endpoint, HttpEndpoint, ReplayConfig},
    gate::GateReport,
    metrics::Run,
    oracle::{self, Fixture},
    record::{ClassPolicy, RecordConfig, RecordServer},
    report::PerformanceReport,
    trace::Trace,
};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("replay") => cmd_replay(&args[1..]),
        Some("canary") => cmd_canary(&args[1..]),
        Some("report") => cmd_report(&args[1..]),
        Some("gate") => cmd_gate(&args[1..]),
        Some("record") => cmd_record(&args[1..]),
        Some("oracle") => cmd_oracle(&args[1..]),
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
  ignis-bench canary --endpoint <url> [--out <canary.json>]
  ignis-bench report --ours <ours.json> --ref <ref.json>
  ignis-bench gate --ours <ours.json> --ref <ref.json> --canary <canary.json> [--out <gate.json>]
  ignis-bench record --listen <proxy> --target <engine-url> --out <load>-trace.jsonl [--class <policy>]
  ignis-bench oracle record --endpoint <url> --artifact <artifact.ninfer> --out <fixture.json> [--max-tokens N]
  ignis-bench oracle compare --fixture <fixture.json> --endpoint <url> --artifact <artifact.ninfer> [--first-n N]
  ignis-bench oracle compare --fixture <fixture.json> --candidate <candidate-fixture.json> [--first-n N]"
    );
}

/// Return the value following the `--key` flag in an argument list, or
/// `None` (the flag is absent, or it has no value). Flags are spelled
/// `--key` in argv (e.g. `--ours <path>`), so the `key` argument is the
/// flag name *without* the `--` prefix.
fn opt(args: &[String], key: &str) -> Option<String> {
    let flag = format!("--{key}");
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag {
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
    let out = opt(args, "out");
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
            if r.consistent() { "" } else { "  <-- DIVERGENT" },
        );
        // An unsane canary carries the sanity reason (the divergence report
        // must say *why*, not just *what*).
        if let Some(reason) = &r.sane_reason {
            println!("             why: {reason}");
        }
    }
    println!("self-consistency: {}", if consistent { "PASS" } else { "FAIL" });
    // Ship the divergence report as JSON (`--out`, spec 02 / ADR 0007).
    if let Some(path) = out {
        let json = serde_json::to_string_pretty(&results).expect("serialize canary results");
        match std::fs::write(&path, &json) {
            Ok(()) => eprintln!("wrote {path}"),
            Err(e) => {
                eprintln!("error: write {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
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

fn cmd_gate(args: &[String]) -> ExitCode {
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
    let canary_path = match require(args, "canary") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let out = opt(args, "out");

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
    // The canary (divergence) results (`canary --out`): the
    // self-consistency half of the v1 verdict (ADR 0007). The v1 gate is
    // the *conjunction* of the performance gate and the self-consistency
    // check (spec 02), so the canary file is required; the performance-
    // only comparison lives in `report`.
    let canary = match load_canaries(&canary_path) {
        Ok(results) => results,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    // The v1 acceptance artifact (ADR 0007): the performance report (the
    // 99% gate, per class) + the divergence report (canary
    // self-consistency), a single overall verdict.
    let artifact = GateReport::new(PerformanceReport::new(&ours, &reference), canary);
    print!("{}", artifact.render());

    // Ship the artifact as JSON (`--out`, spec 02).
    if let Some(path) = out {
        let json =
            serde_json::to_string_pretty(&artifact).expect("serialize the gate artifact");
        match std::fs::write(&path, &json) {
            Ok(()) => eprintln!("wrote {path}"),
            Err(e) => {
                eprintln!("error: write {path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if artifact.passed() {
        ExitCode::SUCCESS
    } else {
        // Which half of the v1 verdict failed (report both when both
        // failed — a bare "the gate failed" would hide one of the two).
        let perf = artifact.performance.gate_passed();
        let consistent = canary::suite_consistent(&artifact.canary);
        let detail = match (perf, consistent) {
            (false, false) => {
                "the 99% performance gate failed AND the self-consistency check failed (a canary diverged)"
            }
            (false, true) => "the 99% performance gate failed",
            (true, false) => "the self-consistency check failed (a canary diverged)",
            (true, true) => unreachable!("gate_passed() && consistent but passed() is false"),
        };
        eprintln!("v1 gate FAILED (ADR 0007): {detail}");
        ExitCode::FAILURE
    }
}

fn cmd_record(args: &[String]) -> ExitCode {
    let listen = match require(args, "listen") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let target = match require(args, "target") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let out = match require(args, "out") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let class = opt(args, "class").unwrap_or_else(|| "first-is-main".into());
    let policy = match ClassPolicy::parse(&class) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    // Pre-flight: the target engine must be reachable and have a model
    // loaded — a clean error beats a proxy that 502s on every request.
    let probe = HttpEndpoint::new(&target);
    match probe.list_models() {
        Ok(models) => eprintln!("target {target}: {}", models.join(", ")),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    }
    let out_display = out.clone();
    let server = match RecordServer::new(RecordConfig {
        listen,
        target,
        out: PathBuf::from(&out),
        class_policy: policy,
    }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = rt.block_on(async move {
        let (url, listener) = server.bind().await?;
        eprintln!(
            "ignis-bench record: capturing on {url} (target {})",
            server.target()
        );
        eprintln!("  agent client -> {url}/v1/chat/completions (OpenAI chat-completions)");
        eprintln!("  session end  -> POST {url}/v1/session/end (or Ctrl-C — the trace is already complete: lines are flushed on arrival)");
        eprintln!("  trace -> {out_display}");
        let summary = server.serve(listener).await?;
        eprintln!(
            "recorded {} requests ({} main, {} sub) over {} ms -> {}",
            summary.requests,
            summary.main,
            summary.sub,
            summary.duration_ms,
            summary
                .file
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(no requests)".into()),
        );
        Ok::<(), String>(())
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `oracle record` / `oracle compare` (P1-04, GitHub #40): dispatch on the
/// oracle sub-subcommand.
fn cmd_oracle(args: &[String]) -> ExitCode {
    match args.first().map(String::as_str) {
        Some("record") => cmd_oracle_record(&args[1..]),
        Some("compare") => cmd_oracle_compare(&args[1..]),
        _ => {
            eprintln!("oracle: expected `record` or `compare`");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

/// Open the artifact's frontend set (`ignis-artifact`) — its `.tokenizer()`
/// is what the recorder/comparer tokenize canary text with. `FrontendSet`
/// owns its parsed tokenizer (not borrowed from the `Reader`), so the
/// reader is dropped once extraction succeeds.
fn open_frontend(artifact_path: &str) -> Result<ignis_artifact::FrontendSet, String> {
    let reader = ignis_artifact::Reader::open(Path::new(artifact_path))
        .map_err(|e| format!("open artifact {artifact_path}: {e}"))?;
    ignis_artifact::FrontendSet::from_reader(&reader)
        .map_err(|e| format!("read the frontend set from {artifact_path}: {e}"))
}

fn cmd_oracle_record(args: &[String]) -> ExitCode {
    let endpoint = match require(args, "endpoint") {
        Ok(u) => u,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let artifact = match require(args, "artifact") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let out = match require(args, "out") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let max_tokens = opt(args, "max-tokens")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(32);

    let frontend = match open_frontend(&artifact) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ep = HttpEndpoint::new(&endpoint);
    let model = match ep.list_models() {
        Ok(models) if !models.is_empty() => {
            eprintln!("engine {endpoint}: {}", models.join(", "));
            models[0].clone()
        }
        Ok(_) => {
            eprintln!("error: engine {endpoint} reports no loaded model");
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let fixture = match oracle::record(&ep, frontend.tokenizer(), model, max_tokens) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    for p in &fixture.prompts {
        println!("recorded {:<16} {} tokens", p.id, p.token_ids.len());
    }
    if let Err(e) = fixture.write(Path::new(&out)) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!("wrote {out}");
    ExitCode::SUCCESS
}

fn cmd_oracle_compare(args: &[String]) -> ExitCode {
    let fixture_path = match require(args, "fixture") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let first_n = opt(args, "first-n").and_then(|v| v.parse::<usize>().ok()).unwrap_or(32);

    let oracle_fixture = match Fixture::read(Path::new(&fixture_path)) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    // The candidate side: either a pre-recorded fixture (`--candidate`, a
    // token list already tokenized the same way) or a live endpoint
    // (`--endpoint` + `--artifact`, recorded fresh through the same
    // recorder).
    let candidate_fixture = if let Some(candidate_path) = opt(args, "candidate") {
        match Fixture::read(Path::new(&candidate_path)) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        let endpoint = match require(args, "endpoint") {
            Ok(u) => u,
            Err(e) => {
                eprintln!("{e}: (or pass --candidate <fixture.json>)");
                print_usage();
                return ExitCode::FAILURE;
            }
        };
        let artifact = match require(args, "artifact") {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e}");
                print_usage();
                return ExitCode::FAILURE;
            }
        };
        let frontend = match open_frontend(&artifact) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        let ep = HttpEndpoint::new(&endpoint);
        if let Err(e) = ep.list_models() {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        match oracle::record(&ep, frontend.tokenizer(), "candidate".into(), oracle_fixture.max_tokens) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    let results = match oracle::compare_fixtures(&oracle_fixture, &candidate_fixture, first_n) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    for r in &results {
        println!(
            "canary {:<16} agreement={:.1}% ({}/{}){}",
            r.id,
            r.agreement * 100.0,
            r.agree,
            r.compared,
            match r.first_divergence {
                Some(pos) => format!("  first divergence @ {pos}"),
                None => String::new(),
            }
        );
    }
    let overall = oracle::overall_agreement(&results);
    println!("overall agreement: {:.1}%", overall * 100.0);
    if overall >= 0.95 {
        ExitCode::SUCCESS
    } else {
        eprintln!("oracle FAILED: overall agreement {:.1}% < 95% (G1 floor)", overall * 100.0);
        ExitCode::FAILURE
    }
}

fn load_run(path: &str) -> Result<Run, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))
}

fn load_canaries(path: &str) -> Result<Vec<CanaryResult>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn opt_reads_the_value_following_a_flag() {
        // Flags are spelled `--key` in argv (the `opt`/`require` seam the
        // subcommands parse with).
        let args = argv(&["--trace", "t.jsonl", "--conc", "4", "--label", "ignis"]);
        assert_eq!(opt(&args, "trace").as_deref(), Some("t.jsonl"));
        assert_eq!(opt(&args, "conc").as_deref(), Some("4"));
        assert_eq!(opt(&args, "label").as_deref(), Some("ignis"));
        // An absent flag is `None` (a default applies, or `require` fails).
        assert_eq!(opt(&args, "out"), None);
    }

    #[test]
    fn require_reads_a_present_flag() {
        let args = argv(&["--ours", "a.json", "--ref", "b.json"]);
        assert_eq!(require(&args, "ours").ok().as_deref(), Some("a.json"));
        assert_eq!(require(&args, "ref").ok().as_deref(), Some("b.json"));
    }

    #[test]
    fn require_rejects_an_absent_flag() {
        let args = argv(&["--ours", "a.json"]);
        assert!(require(&args, "ours").is_ok(), "the flag is present");
        assert!(
            require(&args, "ref").is_err(),
            "an absent flag must be rejected (not silently defaulted)"
        );
    }

    #[test]
    fn a_flag_without_a_value_is_missing() {
        // A trailing `--ref` with no value reads as `None` (the value is
        // missing, not the flag).
        let args = argv(&["--ours", "a.json", "--ref"]);
        assert!(
            opt(&args, "ref").is_none(),
            "a flag at the end of argv has no value"
        );
        assert!(require(&args, "ref").is_err());
    }
}