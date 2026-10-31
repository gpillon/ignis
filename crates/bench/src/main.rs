//! ignis-bench CLI: trace-replay harness + canary suite + v1 gate.
//!
//! Subcommands:
//!   `replay  --trace <trace.jsonl> --endpoint <url> [--conc N] [--scale F]
//!             [--out <run.json>] [--label L] [--as-recorded]
//!             [--retry-engine-full [--retry-max N] [--retry-max-wait S]
//!                                  [--retry-total-wait S]]`
//!       Re-send the JSONL trace against a running endpoint and write the run
//!       (per-request metrics) as JSON. Both options are opt-in; without them
//!       a replay is what every recorded baseline was measured with.
//!       `--as-recorded`: a line whose `prompt` is a recorded chat request
//!       body (a JSON object with `messages`, what a real agent client sent)
//!       is sent as that body: its messages, `tools`, `tool_choice` and every
//!       other field as recorded. The harness still sets `max_tokens` (the
//!       line's), `stream`/`stream_options`, `temperature: 0` and `seed: 0`,
//!       the `class` extension and `enable_thinking` where set, and drops the
//!       recorded `model` (the server's configured model answers). Any other
//!       line is sent as one flat user message, as without the flag. Without
//!       the flag the body is that flat message, so two servers are not
//!       given the same work; baselines recorded that way compare only with
//!       replays that do not use it. The replay says how many lines it sent
//!       as recorded. The body is greedy (`temperature: 0`), so the replay is
//!       reproducible but not the client's own sampling: on a question like
//!       "does MTP pay", its acceptance is an upper bound.
//!       `--retry-engine-full`: a `503` `engine_full` (the engine has no free
//!       lane) is waited out and resent instead of recorded as a failure:
//!       the wait is the response's `Retry-After` seconds (1 s without one),
//!       bounded to `--retry-max-wait` seconds (default 30, at most 60: the
//!       server's `Retry-After` is 1 to 60 s), up to `--retry-max` resends
//!       (default 100) and `--retry-total-wait` seconds of waiting in all
//!       (default 600) per request; the budget is what usually ends a
//!       request that keeps being refused. The run records, per request,
//!       `retries` and `retry_wait_ms` (omitted when zero; the waiting slept,
//!       not the refused round trips); `ttft_ms`/`total_ms` time the attempt
//!       that was admitted, so queueing and service are told apart. Only
//!       `engine_full` is retried. `report` and `gate` print the retries and
//!       the wait per class when there are any.
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
//!   `ttft    --endpoint <url> --artifact <artifact.ninfer> --cells 8192,32768
//!             [--image <screenshot.png> [--image-question <text>]]
//!             [--samples 5] [--max-tokens 8] [--label L] [--profile P]
//!             [--session S] [--corpus <bank.ids>] [--out <record.json>]`
//!       The G2 measurement instrument (P2-05, ADR 0015): time to first
//!       token at an exact prompt length, on cold prefixes, against any
//!       OpenAI-compatible endpoint. Writes one engine's record. `--image`
//!       adds the multimodal cell (GitHub #181): the PNG plus a short
//!       question, after the text cells; with it `--cells` may be omitted.
//!   `g2      --ours <ignis-record.json> --ref <reference-record.json>
//!             [--note <text>] [--out <verdict.json>]`
//!       The G2 verdict over two such records: the ratio of medians per
//!       cell against 1.5. Refuses a verdict when the records are not
//!       live/live (different sessions, a missing cell, a void sample).
//!   `small-card --endpoint <url> [--metrics <url>|off] [--cells 8192,65536]
//!             [--samples 3] [--decode-tokens 256] [--out <small-card.json>]`
//!       The small-card preset (spec 27b-16gb/05, #340): one command against a
//!       running server, a few minutes, ONE JSON to paste into a GitHub
//!       issue. It records the card (name, VRAM, PCIe link and its maximum,
//!       from `nvidia-smi`), the load's VRAM plan lines and shape (the metrics
//!       listener, `/v1/models` with the context, lanes and adaptations the
//!       load runs at, `/v1/models/catalog`, `/v1/config`), decode tok/s on
//!       the coding prompts and on synthetic agent-like turns, thinking off,
//!       and TTFT at the cells (8K and 64K; the prompts are counted through
//!       the server's own `/v1/tokenize`, so no artifact is needed).
//!       `--metrics` defaults to the endpoint's host on the port of the
//!       server's `server.metrics_bind` (`/v1/config`; 9464 when it does not
//!       say); `off` skips it. A piece that cannot be read is `null` with a
//!       line in `notes`. The record carries no credentials and no paths.
//!   `record  --listen <proxy> --target <engine-url> --out <load>-trace.jsonl
//!             [--class <policy>]`
//!       The capture proxy (spec 03): accepts OpenAI chat-completions from a
//!       live agent client, records each request as a trace line, forwards
//!       it to the target engine, and pipes the response back. `POST
//!       /v1/session/end` finalizes the trace and stops the proxy.
//!   `g3      --endpoint <url> --artifact <artifact.ninfer> [--label L]
//!             [--profile P] [--session S] [--corpus <bank.ids>] [--out <record.json>]`
//!       The G3 measurement instrument (P3-07, ADR 0015): the C=1 / C=4 /
//!       ITL cells (spec 03) over HTTP/SSE against any OpenAI-compatible
//!       endpoint. Writes one engine's record.
//!   `g3-gate --ours <ignis-record.json> --ref <reference-record.json>
//!             [--note <text>] [--out <verdict.json>]`
//!       The G3 verdict over two such records: C=1/C=4 aggregate throughput
//!       ratio against 99%, ITL p95 ratio against the reference's own p95.
//!       Refuses a verdict when the records are not live/live (different
//!       sessions, a missing cell, a void/bad sample).
//!   `g4      --endpoint <url> --artifact <artifact.ninfer> --trace <trace.jsonl>
//!             [--label L] [--profile P] [--session S] [--conc N] [--corpus <bank.ids>]
//!             [--out <record.json>]`
//!       The G4 measurement instrument (P4-01, ADR 0015/0021): replays the
//!       load trace and measures the needle-retrieval cells (64K / 128K).
//!       Writes one *launch* of one engine's record — the trace's SHA-256
//!       and the session id ride along, so `g4-gate` can enforce live/live.
//!   `g4-gate --ours <launch1.json> --ours <launch2.json> [...]
//!             --ref <launch1.json> --ref <launch2.json> [...]
//!             [--note <text>] [--out <verdict.json>]`
//!       The pooled G4 verdict over at least two launches per engine
//!       (ADR 0021): the per-class / aggregate pooled throughput ratio
//!       against 99%, and the needle-retrieval floor. Refuses a verdict
//!       when the launches are not live/live (different sessions, different
//!       trace hashes, a missing cell) or fewer than two launches are given
//!       per side.
//!   `g5      --endpoint <url> --artifact <artifact.ninfer> [--label ignis]
//!             [--profile P] [--session S] [--committed-tokens 512]
//!             [--corpus <bank.ids>] [--out <record.json>]`
//!       The G5 measurement instrument (P5-07, ADR 0015/0021): g3's C=1
//!       throughput cell (and its committed-token counter) at three prompt
//!       depths (24K/98K/196K). Writes one *launch* of one engine's record.
//!   `g5-gate --ours <launch1.json> --ours <launch2.json> [...]
//!             --ref <launch1.json> --ref <launch2.json> [...]
//!             [--note <text>] [--out <verdict.json>]`
//!       The pooled G5 verdict over at least two launches per engine
//!       (ADR 0021): one committed-tok/s ratio per depth against 99%,
//!       reusing `g4-gate`'s pooling rule. Refuses a verdict when the
//!       launches are not live/live or fewer than two launches are given
//!       per side.
//!   `g5-equivalence --spec-on <url> --spec-off <url> --artifact <artifact.ninfer>
//!             [--max-tokens 64] [--out <report.json>]`
//!       The phase 5 correctness oracle (P5-07): the canary suite run
//!       greedy against a speculation-on and a speculation-off endpoint (or
//!       the same endpoint restarted between runs), compared token-for-token
//!       for exact equality. Exits non-zero on the first divergence, naming
//!       the canary and the position.
//!   `g5-equivalence capture --endpoint <url> --artifact <artifact.ninfer>
//!             --side spec-on|spec-off --session <id> --out <capture.json>`
//!       One side of that check against the launch live now, kept on disk
//!       with its side and session (#156): two engines do not fit on the card
//!       at once, so the gate captures the spec-off launch, restarts with
//!       speculation on, captures again, and compares with
//!       `--spec-on-capture <on.json> --spec-off-capture <off.json>`.
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
    client::{is_recorded_chat_body, replay, Endpoint, HttpEndpoint, ReplayConfig, RetryPolicy, MAX_RETRY_WAIT},
    equivalence,
    g2, g3, g3_gate, g4, g4_gate, g5, g5_gate,
    gate::GateReport,
    metrics::Run,
    oracle::{self, Fixture},
    record::{ClassPolicy, RecordConfig, RecordServer},
    report::PerformanceReport,
    time::new_session_id,
    trace::Trace,
    small_card,
    ttft::{self, CellSpec, ImageCellSpec, TtftConfig},
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
        Some("ttft") => cmd_ttft(&args[1..]),
        Some("small-card") => cmd_small_card(&args[1..]),
        Some("g2") => cmd_g2(&args[1..]),
        Some("g3") => cmd_g3(&args[1..]),
        Some("g3-gate") => cmd_g3_gate(&args[1..]),
        Some("g4") => cmd_g4(&args[1..]),
        Some("g4-gate") => cmd_g4_gate(&args[1..]),
        Some("g5") => cmd_g5(&args[1..]),
        Some("g5-gate") => cmd_g5_gate(&args[1..]),
        Some("g5-equivalence") => cmd_g5_equivalence(&args[1..]),
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
                     [--as-recorded] [--retry-engine-full [--retry-max N] [--retry-max-wait S] [--retry-total-wait S]]
  ignis-bench canary --endpoint <url> [--out <canary.json>]
  ignis-bench report --ours <ours.json> --ref <ref.json>
  ignis-bench gate --ours <ours.json> --ref <ref.json> --canary <canary.json> [--out <gate.json>]
  ignis-bench record --listen <proxy> --target <engine-url> --out <load>-trace.jsonl [--class <policy>]
  ignis-bench oracle record --endpoint <url> --artifact <artifact.ninfer> --out <fixture.json> [--max-tokens N]
  ignis-bench oracle compare --fixture <fixture.json> --endpoint <url> --artifact <artifact.ninfer> [--first-n N]
  ignis-bench oracle compare --fixture <fixture.json> --candidate <candidate-fixture.json> [--first-n N]
  ignis-bench ttft --endpoint <url> --artifact <artifact.ninfer> --cells 8192,32768 [--samples 5] [--max-tokens 8]
                   [--image <screenshot.png> [--image-question <text>]]
                   [--label ignis] [--profile <text>] [--session <id>] [--corpus <bank.ids>] [--out <record.json>]
  ignis-bench small-card --endpoint <url> [--metrics <url>|off] [--cells 8192,65536] [--samples 3]
                         [--decode-tokens 256] [--out <small-card.json>]
  ignis-bench g2 --ours <ignis-record.json> --ref <reference-record.json> [--note <text>] [--out <verdict.json>]
  ignis-bench g3 --endpoint <url> --artifact <artifact.ninfer> [--label ignis] [--profile <text>]
                 [--session <id>] [--corpus <bank.ids>] [--out <record.json>]
  ignis-bench g3-gate --ours <ignis-record.json> --ref <reference-record.json> [--note <text>] [--out <verdict.json>]
  ignis-bench g4 --endpoint <url> --artifact <artifact.ninfer> --trace <trace.jsonl> [--label ignis]
                 [--profile <text>] [--session <id>] [--conc N] [--corpus <bank.ids>] [--out <record.json>]
  ignis-bench g4-gate --ours <launch.json> [--ours <launch2.json> ...] --ref <launch.json> [--ref <launch2.json> ...]
                      [--note <text>] [--out <verdict.json>]
  ignis-bench g5 --endpoint <url> --artifact <artifact.ninfer> [--label ignis] [--profile <text>]
                 [--session <id>] [--committed-tokens 512] [--corpus <bank.ids>] [--out <record.json>]
  ignis-bench g5-gate --ours <launch.json> [--ours <launch2.json> ...] --ref <launch.json> [--ref <launch2.json> ...]
                      [--note <text>] [--out <verdict.json>]
  ignis-bench g5-equivalence (--spec-on <url> | --spec-on-capture <capture.json>)
                      (--spec-off <url> | --spec-off-capture <capture.json>)
                      [--artifact <artifact.ninfer> (required for a live side)] [--session <id>]
                      [--max-tokens 64] [--out <report.json>]
  ignis-bench g5-equivalence capture --endpoint <url> --artifact <artifact.ninfer>
                      --side spec-on|spec-off --session <id> --out <capture.json> [--max-tokens 64]
    two launches, one card: capture --side spec-off, restart with speculation on,
    capture --side spec-on, then --spec-on-capture <on.json> --spec-off-capture <off.json>

environment:
  {timeout_env}=<seconds>   the per-request deadline every subcommand drives an engine with
                            (default {timeout_default} s; 0 removes it). A measurement request
                            that outlives it is reported as the client's own timeout, not as an
                            engine failure.",
        timeout_env = ignis_bench::client::REQUEST_TIMEOUT_ENV,
        timeout_default = ignis_bench::client::DEFAULT_REQUEST_TIMEOUT.as_secs(),
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

/// Build the endpoint for a subcommand and **say what deadline it drives
/// with** (GitHub #138).
///
/// The deadline that made the G4 needle cell unmeasurable was invisible:
/// nothing configured it, nothing printed it, and nothing in the run record
/// named it, so four launches across two engines read its symptom as an
/// engine fault. A run that prints the number it is using cannot cost
/// anyone that again.
fn endpoint_for(base_url: &str) -> HttpEndpoint {
    let ep = HttpEndpoint::new(base_url);
    match ep.request_timeout() {
        Some(deadline) => eprintln!(
            "request deadline: {} s per request ({}=0 removes it)",
            deadline.as_secs(),
            ignis_bench::client::REQUEST_TIMEOUT_ENV
        ),
        None => eprintln!("request deadline: none (waiting as long as the engine takes)"),
    }
    ep
}

/// `replay`'s opt-in fidelity options: `--as-recorded`, and
/// `--retry-engine-full` with its `--retry-max` / `--retry-max-wait` /
/// `--retry-total-wait` bounds. A bound without the switch it tunes is
/// refused rather than ignored, and so is a wait that is not a sane number
/// of seconds (or a `--retry-max-wait` past the server's 60 s).
fn replay_options(args: &[String]) -> Result<(bool, Option<RetryPolicy>), String> {
    let as_recorded = args.iter().any(|a| a == "--as-recorded");
    let defaults = RetryPolicy::default();
    let retry_max = opt(args, "retry-max");
    let retry_max_wait = opt(args, "retry-max-wait");
    let retry_total_wait = opt(args, "retry-total-wait");
    if !args.iter().any(|a| a == "--retry-engine-full") {
        if retry_max.is_some() || retry_max_wait.is_some() || retry_total_wait.is_some() {
            return Err("--retry-max, --retry-max-wait and --retry-total-wait need --retry-engine-full".into());
        }
        return Ok((as_recorded, None));
    }
    let max_retries = match retry_max {
        Some(v) => v.parse::<u32>().map_err(|_| format!("--retry-max {v}: not a whole number"))?,
        None => defaults.max_retries,
    };
    let seconds = |flag: &str, v: &str| {
        v.parse::<f64>()
            .ok()
            .and_then(|s| std::time::Duration::try_from_secs_f64(s).ok())
            .ok_or_else(|| format!("--{flag} {v}: not a number of seconds"))
    };
    let max_wait = match retry_max_wait {
        Some(v) => {
            let wait = seconds("retry-max-wait", &v)?;
            if wait > MAX_RETRY_WAIT {
                return Err(format!(
                    "--retry-max-wait {v}: the server's Retry-After is 1 to 60 seconds, so a longer cap waits for nothing"
                ));
            }
            wait
        }
        None => defaults.max_wait,
    };
    let total_wait = match retry_total_wait {
        Some(v) => seconds("retry-total-wait", &v)?,
        None => defaults.total_wait,
    };
    Ok((as_recorded, Some(RetryPolicy { max_retries, max_wait, total_wait })))
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

    let (as_recorded, retry) = match replay_options(args) {
        Ok(options) => options,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut ep = endpoint_for(&endpoint).as_recorded(as_recorded);
    if let Some(policy) = retry {
        ep = ep.with_retry(policy);
    }
    if as_recorded {
        let recorded = trace.lines().iter().filter(|l| is_recorded_chat_body(&l.prompt)).count();
        eprintln!(
            "as recorded: {recorded} of {} lines sent as recorded (a line holding a chat request body is sent as that body, any other as one flat message)",
            trace.len()
        );
    }
    if let Some(policy) = retry {
        eprintln!(
            "engine_full: waited out and resent (up to {} times, each wait at most {} s, {} s of waiting in all)",
            policy.max_retries,
            policy.max_wait.as_secs_f64(),
            policy.total_wait.as_secs_f64()
        );
    }
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
    let ep = endpoint_for(&endpoint);
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
            "canary {:<12} sane={} deterministic={} answer={}{}",
            r.id,
            r.sane,
            r.deterministic,
            // GitHub #144: an empty answer because the turn was still
            // thinking is a budget fact, and the report says so rather than
            // leaving it to be inferred from an empty field.
            if r.thinking_only() {
                "thinking-only"
            } else {
                "reached"
            },
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
    print!("{}{}{}", report.render(), ours.queueing_note(), reference.queueing_note());
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
    print!("{}{}{}", artifact.render(), ours.queueing_note(), reference.queueing_note());

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
    let probe = endpoint_for(&target);
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

/// Every value following a repeated `--key` flag (e.g. `--note`), in argv
/// order. [`opt`] takes only the first; the G2 verdict may carry several
/// known inequalities.
fn repeated_opt(args: &[String], key: &str) -> Vec<String> {
    let flag = format!("--{key}");
    args.windows(2)
        .filter(|w| w[0] == flag)
        .map(|w| w[1].clone())
        .collect()
}

/// `ttft` (P2-05, GitHub #87): measure time to first token at one or more
/// exact prompt lengths, on cold prefixes, against one OpenAI-compatible
/// endpoint. The record it writes is one half of a G2 verdict; `g2`
/// compares two of them.
fn cmd_ttft(args: &[String]) -> ExitCode {
    // GitHub #181: a record may carry only the multimodal cell.
    let image_path = opt(args, "image").map(PathBuf::from);
    let cells_required = if image_path.is_some() {
        Ok(opt(args, "cells").unwrap_or_default())
    } else {
        require(args, "cells")
    };
    let (endpoint, artifact, cells_raw) = match (
        require(args, "endpoint"),
        require(args, "artifact"),
        cells_required,
    ) {
        (Ok(e), Ok(a), Ok(c)) => (e, a, c),
        (e, a, c) => {
            for err in [e.err(), a.err(), c.err()].into_iter().flatten() {
                eprintln!("{err}");
            }
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let samples = opt(args, "samples")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(ttft::DEFAULT_SAMPLES);
    let max_tokens = opt(args, "max-tokens")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(ttft::DEFAULT_MAX_TOKENS);
    let label = opt(args, "label").unwrap_or_else(|| "ignis".into());
    let profile = opt(args, "profile").unwrap_or_else(|| "unrecorded".into());
    // Live/live (ADR 0015): the *same* session id must be passed to both
    // engines' runs, so a generated one is announced loudly enough to be
    // reused by the second run.
    let session = opt(args, "session").unwrap_or_else(new_session_id);
    let out = opt(args, "out");
    let corpus = opt(args, "corpus").map(PathBuf::from);
    if let Some(path) = &corpus {
        // Fail before any request is sent: a missing corpus file would
        // otherwise fail every cell (and the gate) with a late error.
        if !path.is_file() {
            eprintln!("error: --corpus file not found: {path:?}");
            return ExitCode::FAILURE;
        }
        eprintln!("corpus mode: prompts are cut from {path:?} (detokenized windows, no filler growth)");
    }

    let mut specs = Vec::new();
    for raw in cells_raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match raw.parse::<u32>() {
            Ok(prompt_tokens) => specs.push(CellSpec { prompt_tokens, samples }),
            Err(_) => {
                eprintln!("error: --cells expects comma-separated token counts, got `{raw}`");
                return ExitCode::FAILURE;
            }
        }
    }
    if specs.is_empty() && image_path.is_none() {
        eprintln!("error: --cells named no cells");
        return ExitCode::FAILURE;
    }
    let image = match &image_path {
        None => None,
        Some(path) => match std::fs::read(path) {
            Ok(png) => Some(ImageCellSpec {
                png,
                question: opt(args, "image-question").unwrap_or_else(|| ttft::DEFAULT_IMAGE_QUESTION.into()),
                samples,
            }),
            Err(e) => {
                eprintln!("error: --image {path:?}: {e}");
                return ExitCode::FAILURE;
            }
        },
    };

    // The artifact's own tokenizer + chat template: what makes a cell's
    // claimed prompt length the length the engine actually prefills.
    let frontend = match open_frontend(&artifact) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ep = endpoint_for(&endpoint);
    let engine = match ep.list_models() {
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
    eprintln!(
        "ttft session {session}: {} cells x {samples} samples (+1 warmup) against {label}",
        specs.len() + usize::from(image.is_some())
    );

    let cfg = TtftConfig {
        cells: specs,
        max_tokens,
        label,
        profile,
        artifact,
        session,
        corpus,
        image,
    };
    let record = ttft::measure(&ep, &frontend, engine, endpoint, &cfg);
    print!("{}", record.render());

    if let Some(path) = &out {
        if let Err(e) = record.write(Path::new(path)) {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    // A cell whose samples were not all cold measured something other than
    // prefill; the gate check would refuse it anyway, so say so here.
    let contaminated: Vec<u32> = record
        .cells
        .iter()
        .filter(|c| !c.all_cold())
        .map(|c| c.prompt_tokens)
        .collect();
    if contaminated.is_empty() {
        eprintln!("all samples cold; pass --session {} to the other engine's run", cfg.session);
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "error: cells {contaminated:?} did not produce a full set of cold samples — this \
             record cannot decide the gate"
        );
        ExitCode::FAILURE
    }
}

/// `small-card` (#340, spec 27b-16gb/05 Part B): the one-command record a
/// 16 GB card's owner pastes into an issue. Prints the JSON on stdout (the
/// progress goes to stderr), and writes it to `--out` too when given.
fn cmd_small_card(args: &[String]) -> ExitCode {
    let endpoint = match require(args, "endpoint") {
        Ok(e) => e,
        Err(e) => {
            eprintln!("{e}");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let metrics = match opt(args, "metrics").as_deref() {
        Some("off") => small_card::MetricsAt::Off,
        Some(url) => small_card::MetricsAt::Url(url.to_string()),
        None => small_card::MetricsAt::Server,
    };
    let mut cells = Vec::new();
    match opt(args, "cells") {
        None => cells.extend(small_card::DEFAULT_CELLS),
        Some(raw) => {
            for part in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                match part.parse::<u32>() {
                    Ok(n) => cells.push(n),
                    Err(_) => {
                        eprintln!("error: --cells expects comma-separated token counts, got `{part}`");
                        return ExitCode::FAILURE;
                    }
                }
            }
        }
    }
    let whole = |key: &str, default: usize| -> Result<usize, String> {
        match opt(args, key) {
            None => Ok(default),
            Some(v) => v.parse::<usize>().ok().filter(|&n| n > 0).ok_or_else(|| format!("--{key} {v}: not a positive whole number")),
        }
    };
    let (samples, decode_tokens) = match (whole("samples", small_card::DEFAULT_SAMPLES), whole("decode-tokens", small_card::DEFAULT_DECODE_TOKENS as usize)) {
        (Ok(s), Ok(d)) => (s, d as u32),
        (s, d) => {
            for err in [s.err(), d.err()].into_iter().flatten() {
                eprintln!("error: {err}");
            }
            return ExitCode::FAILURE;
        }
    };
    let opts = small_card::Options { endpoint, metrics, cells, samples, decode_tokens };
    let report = small_card::run(&opts, &small_card::run_nvidia_smi);
    let json = match small_card::to_json(&report) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    print!("{json}");
    if let Some(path) = opt(args, "out") {
        if let Err(e) = std::fs::write(&path, &json) {
            eprintln!("error: {path}: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    if report.server.model_id.is_none() {
        eprintln!("error: no model answered at {}/v1/models: nothing was measured", report.endpoint);
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// `g2` (P2-05, GitHub #87): the G2 verdict over two TTFT records. Refuses
/// outright — rather than reporting a failure — when the two records do not
/// support a live/live comparison (ADR 0015).
fn cmd_g2(args: &[String]) -> ExitCode {
    let (ours_path, ref_path) = match (require(args, "ours"), require(args, "ref")) {
        (Ok(o), Ok(r)) => (o, r),
        (o, r) => {
            for err in [o.err(), r.err()].into_iter().flatten() {
                eprintln!("{err}");
            }
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let ours = match ttft::Record::read(Path::new(&ours_path)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let reference = match ttft::Record::read(Path::new(&ref_path)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut verdict = match g2::check(&ours, &reference) {
        Ok(v) => v,
        Err(refusal) => {
            eprintln!("G2 gate check REFUSED: {refusal}");
            return ExitCode::FAILURE;
        }
    };
    verdict.notes.extend(repeated_opt(args, "note"));
    print!("{}", verdict.render());
    if let Some(path) = opt(args, "out") {
        if let Err(e) = verdict.write(Path::new(&path)) {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    if verdict.passed {
        ExitCode::SUCCESS
    } else {
        eprintln!("G2 gate FAILED (ADR 0015: ratio <= {} per cell)", verdict.threshold);
        ExitCode::FAILURE
    }
}

/// `g3` (P3-07, GitHub #100): measure the C=1 / C=4 / ITL cells (spec 03)
/// against one OpenAI-compatible endpoint. The record it writes is one half
/// of a G3 verdict; `g3-gate` compares two of them.
fn cmd_g3(args: &[String]) -> ExitCode {
    let (endpoint, artifact) = match (require(args, "endpoint"), require(args, "artifact")) {
        (Ok(e), Ok(a)) => (e, a),
        (e, a) => {
            for err in [e.err(), a.err()].into_iter().flatten() {
                eprintln!("{err}");
            }
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let label = opt(args, "label").unwrap_or_else(|| "ignis".into());
    let profile = opt(args, "profile").unwrap_or_else(|| "unrecorded".into());
    // Live/live (ADR 0015): the *same* session id must be passed to both
    // engines' runs.
    let session = opt(args, "session").unwrap_or_else(new_session_id);
    let out = opt(args, "out");
    let corpus = opt(args, "corpus").map(PathBuf::from);
    if let Some(path) = &corpus {
        // Fail before any request is sent: a missing corpus file would
        // otherwise fail every cell (and the gate) with a late error.
        if !path.is_file() {
            eprintln!("error: --corpus file not found: {path:?}");
            return ExitCode::FAILURE;
        }
        eprintln!(
            "corpus mode: prompts are cut from {path:?} (detokenized windows, no filler growth — \
             needed at the ITL cell's 32,768-token prefiller scale)"
        );
    }

    let frontend = match open_frontend(&artifact) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ep = endpoint_for(&endpoint);
    let engine = match ep.list_models() {
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
    // GitHub #426: say which EOS control the ITL lanes carry, since ignis
    // refuses the reference's.
    match ep.eos_control() {
        Ok(control) => {
            let eos = ttft::PromptTemplate::eos_token_ids(&frontend);
            eprintln!("ITL decode lanes suppress EOS with {}", control.apply(serde_json::json!({}), &eos));
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    }
    eprintln!("g3 session {session}: C=1 / C=4 / ITL against {label}");

    let cfg = g3::G3Config {
        label,
        profile,
        artifact,
        session,
        throughput: g3::ThroughputSpec::default(),
        itl: g3::ItlConfig::default(),
        corpus,
    };
    let record = g3::measure(&ep, &frontend, engine, endpoint, &cfg);
    print!("{}", record.render());

    if let Some(path) = &out {
        if let Err(e) = record.write(Path::new(path)) {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    let contaminated = !record.c1.all_cold() || !record.c4.all_cold() || !record.itl.all_cold();
    if contaminated {
        eprintln!(
            "error: at least one cell did not produce a full set of cold/complete samples — \
             this record cannot decide the gate"
        );
        ExitCode::FAILURE
    } else {
        eprintln!("all cells cold; pass --session {} to the other engine's run", cfg.session);
        ExitCode::SUCCESS
    }
}

/// `g3-gate` (P3-07, GitHub #100): the G3 verdict over two records. Refuses
/// outright — rather than reporting a failure — when the two records do not
/// support a live/live comparison (ADR 0015).
fn cmd_g3_gate(args: &[String]) -> ExitCode {
    let (ours_path, ref_path) = match (require(args, "ours"), require(args, "ref")) {
        (Ok(o), Ok(r)) => (o, r),
        (o, r) => {
            for err in [o.err(), r.err()].into_iter().flatten() {
                eprintln!("{err}");
            }
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let ours = match g3::Record::read(Path::new(&ours_path)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let reference = match g3::Record::read(Path::new(&ref_path)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut verdict = match g3_gate::check(&ours, &reference) {
        Ok(v) => v,
        Err(refusal) => {
            eprintln!("G3 gate check REFUSED: {refusal}");
            return ExitCode::FAILURE;
        }
    };
    verdict.notes.extend(repeated_opt(args, "note"));
    print!("{}", verdict.render());
    if let Some(path) = opt(args, "out") {
        if let Err(e) = verdict.write(Path::new(&path)) {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    if verdict.passed {
        if let Some(warning) = &verdict.itl.warning {
            eprintln!("G3 gate PASSED, but flagged: {warning}");
        }
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "G3 gate FAILED (ADR 0015: tok/s ratio >= {} on C=1/C=4, ITL p95 ratio <= {} clean / \
             <= {} tolerated-with-warning)",
            verdict.throughput_threshold, verdict.itl_pass_threshold, verdict.itl_fail_threshold
        );
        ExitCode::FAILURE
    }
}

/// `g4` (P4-01, GitHub #117): replay the load trace and measure the
/// needle-retrieval cells against one OpenAI-compatible endpoint. The
/// record it writes is one *launch* of one engine; `g4-gate` pools at
/// least two of them per engine (ADR 0021) into a verdict.
fn cmd_g4(args: &[String]) -> ExitCode {
    let (endpoint, artifact, trace_path) = match (
        require(args, "endpoint"),
        require(args, "artifact"),
        require(args, "trace"),
    ) {
        (Ok(e), Ok(a), Ok(t)) => (e, a, t),
        (e, a, t) => {
            for err in [e.err(), a.err(), t.err()].into_iter().flatten() {
                eprintln!("{err}");
            }
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let label = opt(args, "label").unwrap_or_else(|| "ignis".into());
    let profile = opt(args, "profile").unwrap_or_else(|| "unrecorded".into());
    // Live/live (ADR 0015): the *same* session id must be passed to every
    // launch of both engines.
    let session = opt(args, "session").unwrap_or_else(new_session_id);
    let out = opt(args, "out");
    let conc = opt(args, "conc").and_then(|v| v.parse::<usize>().ok()).unwrap_or(8);
    let corpus = opt(args, "corpus").map(PathBuf::from);
    if let Some(path) = &corpus {
        // Fail before any request is sent: a missing corpus file would
        // otherwise fail every needle cell with a late error.
        if !path.is_file() {
            eprintln!("error: --corpus file not found: {path:?}");
            return ExitCode::FAILURE;
        }
        eprintln!(
            "corpus mode: needle haystacks are cut from {path:?} (bounded generation, no filler \
             growth — needed at the needle cell's 64K/128K scale)"
        );
    }

    let trace_bytes = match std::fs::read(&trace_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: read {trace_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    // The bytes are hashed as-is (`trace_sha256`), so parsing must fail
    // loudly on invalid UTF-8 rather than lossily substitute — a mangled
    // trace would parse to something other than the bytes the hash
    // actually names (`Trace::from_path`'s `read_to_string` has the same
    // strictness; this path keeps the raw bytes around for the hash, so it
    // cannot reuse that helper directly).
    let trace_text = match String::from_utf8(trace_bytes.clone()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {trace_path} is not valid UTF-8: {e}");
            return ExitCode::FAILURE;
        }
    };
    let trace = match Trace::from_jsonl(&trace_text) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
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
    let ep = endpoint_for(&endpoint);
    let engine = match ep.list_models() {
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
    eprintln!("g4 session {session}: trace replay + needle retrieval against {label}");

    let ep: Arc<dyn Endpoint> = Arc::new(ep);
    let cfg = g4::G4Config {
        label,
        profile,
        artifact,
        session,
        needle_context_tokens: vec![g4::NEEDLE_CONTEXT_64K, g4::NEEDLE_CONTEXT_128K],
        replay: ReplayConfig { max_concurrency: conc, time_scale: 1.0 },
        corpus,
    };
    let record = g4::measure(ep, &frontend, engine, endpoint, &trace, &trace_bytes, &cfg);
    print!("{}", record.render());

    if let Some(path) = &out {
        if let Err(e) = record.write(Path::new(path)) {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    let all_needles_ok = record.needles.iter().all(|n| n.passed());
    if all_needles_ok {
        eprintln!("all needle cells retrieved; pass --session {} to the other launches", cfg.session);
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "error: at least one needle-retrieval cell missed or failed — this record still \
             writes, but a G4 verdict built from it will fail that floor"
        );
        ExitCode::FAILURE
    }
}

/// `g4-gate` (P4-01, GitHub #117): the pooled G4 verdict over at least two
/// launches per engine (ADR 0021). Refuses outright — rather than
/// reporting a failure — when the launches do not support a live/live
/// comparison (different sessions, different trace hashes, a missing
/// cell).
fn cmd_g4_gate(args: &[String]) -> ExitCode {
    let ours_paths = repeated_opt(args, "ours");
    let ref_paths = repeated_opt(args, "ref");
    if ours_paths.is_empty() || ref_paths.is_empty() {
        eprintln!("--ours and --ref must each be passed at least once (repeat the flag per launch)");
        print_usage();
        return ExitCode::FAILURE;
    }
    let load = |paths: &[String]| -> Result<Vec<g4::Record>, String> {
        paths.iter().map(|p| g4::Record::read(Path::new(p))).collect()
    };
    let ours = match load(&ours_paths) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let reference = match load(&ref_paths) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut verdict = match g4_gate::check(&ours, &reference) {
        Ok(v) => v,
        Err(refusal) => {
            eprintln!("G4 gate check REFUSED: {refusal}");
            return ExitCode::FAILURE;
        }
    };
    verdict.notes.extend(repeated_opt(args, "note"));
    print!("{}", verdict.render());
    if let Some(path) = opt(args, "out") {
        if let Err(e) = verdict.write(Path::new(&path)) {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    if verdict.passed {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "G4 gate FAILED (ADR 0021: pooled tok/s ratio >= {} per class and aggregate, every \
             needle cell retrieved)",
            verdict.threshold
        );
        ExitCode::FAILURE
    }
}

/// `g5` (P5-07, GitHub #151): g3's C=1 throughput cell (and its
/// committed-token counter), measured at three prompt depths (24K/98K/196K)
/// against one OpenAI-compatible endpoint. The record it writes is one
/// *launch* of one engine; `g5-gate` pools at least two of them per engine
/// (ADR 0021) into a per-depth verdict.
fn cmd_g5(args: &[String]) -> ExitCode {
    let (endpoint, artifact) = match (require(args, "endpoint"), require(args, "artifact")) {
        (Ok(e), Ok(a)) => (e, a),
        (e, a) => {
            for err in [e.err(), a.err()].into_iter().flatten() {
                eprintln!("{err}");
            }
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let label = opt(args, "label").unwrap_or_else(|| "ignis".into());
    let profile = opt(args, "profile").unwrap_or_else(|| "unrecorded".into());
    // Live/live (ADR 0015): the *same* session id must be passed to every
    // launch of both engines.
    let session = opt(args, "session").unwrap_or_else(new_session_id);
    let committed_tokens =
        opt(args, "committed-tokens").and_then(|v| v.parse::<u32>().ok()).unwrap_or(g5::COMMITTED_TOKENS);
    let out = opt(args, "out");
    let corpus = opt(args, "corpus").map(PathBuf::from);
    if let Some(path) = &corpus {
        // Fail before any request is sent: a missing corpus file would
        // otherwise fail every depth cell with a late error.
        if !path.is_file() {
            eprintln!("error: --corpus file not found: {path:?}");
            return ExitCode::FAILURE;
        }
        eprintln!(
            "corpus mode: prompts are cut from {path:?} (detokenized windows, no filler growth — \
             needed at the 24K-196K depth scale)"
        );
    }

    let frontend = match open_frontend(&artifact) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ep = endpoint_for(&endpoint);
    let engine = match ep.list_models() {
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
    eprintln!("g5 session {session}: committed tok/s @24K/98K/196K against {label}");

    let cfg = g5::G5Config { label, profile, artifact, session, committed_tokens, corpus };
    let record = g5::measure(&ep, &frontend, engine, endpoint, &cfg);
    print!("{}", record.render());

    if let Some(path) = &out {
        if let Err(e) = record.write(Path::new(path)) {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    if record.all_cold() {
        eprintln!("all depth cells cold; pass --session {} to the other engine's run", cfg.session);
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "error: at least one depth cell did not produce a full set of cold/complete samples — \
             this record cannot decide the gate"
        );
        ExitCode::FAILURE
    }
}

/// `g5-gate` (P5-07, GitHub #151): the pooled G5 verdict over at least two
/// launches per engine (ADR 0021), one ratio per depth. Refuses outright —
/// rather than reporting a failure — when the launches do not support a
/// live/live comparison (different sessions, a missing depth cell) or fewer
/// than two launches are given per side.
fn cmd_g5_gate(args: &[String]) -> ExitCode {
    let ours_paths = repeated_opt(args, "ours");
    let ref_paths = repeated_opt(args, "ref");
    if ours_paths.is_empty() || ref_paths.is_empty() {
        eprintln!("--ours and --ref must each be passed at least once (repeat the flag per launch)");
        print_usage();
        return ExitCode::FAILURE;
    }
    let load = |paths: &[String]| -> Result<Vec<g5::Record>, String> {
        paths.iter().map(|p| g5::Record::read(Path::new(p))).collect()
    };
    let ours = match load(&ours_paths) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let reference = match load(&ref_paths) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut verdict = match g5_gate::check(&ours, &reference) {
        Ok(v) => v,
        Err(refusal) => {
            eprintln!("G5 gate check REFUSED: {refusal}");
            return ExitCode::FAILURE;
        }
    };
    verdict.notes.extend(repeated_opt(args, "note"));
    print!("{}", verdict.render());
    if let Some(path) = opt(args, "out") {
        if let Err(e) = verdict.write(Path::new(&path)) {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    if verdict.passed {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "G5 gate FAILED (ADR 0021: pooled committed tok/s ratio >= {} at every depth)",
            verdict.threshold
        );
        ExitCode::FAILURE
    }
}

/// Where one side of `g5-equivalence` comes from: a live endpoint, or a
/// capture file an earlier launch wrote (`g5-equivalence capture`, #156).
#[derive(Debug, Clone, PartialEq)]
enum EquivalenceSide {
    Endpoint(String),
    Capture(String),
}

/// Resolve `--<side> <url>` / `--<side>-capture <file>`: exactly one of the
/// two must be given.
fn equivalence_side(args: &[String], side: &str) -> Result<EquivalenceSide, String> {
    let capture_key = format!("{side}-capture");
    match (opt(args, side), opt(args, &capture_key)) {
        (Some(url), None) => Ok(EquivalenceSide::Endpoint(url)),
        (None, Some(path)) => Ok(EquivalenceSide::Capture(path)),
        _ => Err(format!(
            "exactly one of --{side} <url> or --{capture_key} <capture.json> is required"
        )),
    }
}

/// Take one side's capture from a live endpoint: check it answers, then run
/// the canaries through it.
fn capture_endpoint(
    url: &str,
    side: equivalence::Side,
    session: &str,
    artifact: Option<&str>,
    max_tokens: u32,
) -> Result<equivalence::Capture, String> {
    let side_name = side.as_str();
    let artifact = artifact
        .ok_or_else(|| format!("--artifact is required to capture {side_name} from {url}"))?;
    let frontend = open_frontend(artifact)?;
    let endpoint = endpoint_for(url);
    endpoint.list_models().map_err(|e| format!("{side_name} endpoint {url}: {e}"))?;
    equivalence::capture(&endpoint, frontend.tokenizer(), max_tokens, side, session)
        .map_err(|e| format!("{side_name}: {e}"))
}

/// `g5-equivalence capture --endpoint <url> --artifact <a> --out <file>`:
/// one side of the check against the launch that is live now (#156).
fn cmd_g5_equivalence_capture(args: &[String]) -> ExitCode {
    let side = require(args, "side").and_then(|s| equivalence::Side::parse(&s));
    let (url, out, side, session) =
        match (require(args, "endpoint"), require(args, "out"), side, require(args, "session")) {
            (Ok(u), Ok(o), Ok(s), Ok(n)) => (u, o, s, n),
            (u, o, s, n) => {
                for err in [u.err(), o.err(), s.err(), n.err()].into_iter().flatten() {
                    eprintln!("{err}");
                }
                print_usage();
                return ExitCode::FAILURE;
            }
        };
    let max_tokens = opt(args, "max-tokens")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(equivalence::EQUIVALENCE_MAX_TOKENS);
    let artifact = opt(args, "artifact");
    let captured = match capture_endpoint(&url, side, &session, artifact.as_deref(), max_tokens) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = captured.write(Path::new(&out)) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!("wrote {out} ({} canaries, max_tokens {})", captured.canaries.len(), max_tokens);
    ExitCode::SUCCESS
}

/// `g5-equivalence` (P5-07, GitHub #151): the phase 5 correctness oracle —
/// the canary suite run greedy with speculation on and off, each side a live
/// endpoint or a capture an earlier launch wrote (#156), compared
/// token-for-token for exact equality. Exits non-zero on the first
/// divergence, naming the canary and the position.
fn cmd_g5_equivalence(args: &[String]) -> ExitCode {
    if args.first().map(String::as_str) == Some("capture") {
        return cmd_g5_equivalence_capture(&args[1..]);
    }
    let (on_side, off_side) = match (
        equivalence_side(args, "spec-on"),
        equivalence_side(args, "spec-off"),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            for err in [a.err(), b.err()].into_iter().flatten() {
                eprintln!("{err}");
            }
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let max_tokens = opt(args, "max-tokens")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(equivalence::EQUIVALENCE_MAX_TOKENS);
    let out = opt(args, "out");
    let artifact = opt(args, "artifact");
    let session = opt(args, "session").unwrap_or_else(new_session_id);
    let both_captures = matches!(
        (&on_side, &off_side),
        (EquivalenceSide::Capture(_), EquivalenceSide::Capture(_))
    );
    if both_captures && opt(args, "max-tokens").is_some() {
        eprintln!("note: --max-tokens is ignored: both sides are captures, which carry their own budget");
    }

    let load = |source: &EquivalenceSide, side: equivalence::Side| match source {
        EquivalenceSide::Endpoint(url) => {
            capture_endpoint(url, side, &session, artifact.as_deref(), max_tokens)
        }
        EquivalenceSide::Capture(path) => equivalence::Capture::read(Path::new(path)),
    };
    let results = match load(&on_side, equivalence::Side::SpecOn)
        .and_then(|on| load(&off_side, equivalence::Side::SpecOff).map(|off| (on, off)))
        .and_then(|(on, off)| equivalence::compare_captures(&on, &off))
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    for r in &results {
        println!(
            "canary {:<16} {}{}",
            r.id,
            if r.equivalent() { "EQUIVALENT" } else { "DIVERGENT" },
            match r.first_divergence {
                Some(pos) => format!("  first divergence @ {pos}"),
                None => String::new(),
            }
        );
    }
    if let Some(path) = &out {
        let json = match serde_json::to_string_pretty(&results) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("error: serialize the equivalence report: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = std::fs::write(path, json) {
            eprintln!("error: write {path}: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!("wrote {path}");
    }
    match equivalence::first_failure(&results) {
        None => {
            eprintln!("all canaries equivalent: greedy spec-on == spec-off");
            ExitCode::SUCCESS
        }
        Some((id, pos)) => {
            eprintln!("error: canary {id} diverged at position {pos}: speculation changed greedy output");
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
    let ep = endpoint_for(&endpoint);
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
        let ep = endpoint_for(&endpoint);
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
    println!("overall free-running agreement: {:.1}%", overall * 100.0);
    // ADR 0014: free-running comparison is a DIAGNOSTIC, not the G1 gate. A
    // single divergence changes the candidate's own prefix and decorrelates
    // every later position, so a low figure here means "the continuations
    // parted ways", not "the forward pass is broken". The G1 correctness
    // floor is teacher-forced next-token agreement
    // (`oracle::score_teacher_forced`), driven by the GPU-profile test
    // `crates/server/tests/oracle_teacher_forced_gpu.rs`. So this command
    // reports and never fails the process on the agreement figure.
    println!(
        "note: free-running agreement is informational only (ADR 0014). The G1 \
         correctness floor is teacher-forced next-token agreement >= {:.0}%, \
         measured by the GPU profile.",
        oracle::G1_AGREEMENT_FLOOR * 100.0
    );
    ExitCode::SUCCESS
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
    fn replay_fidelity_options_are_opt_in_with_bounded_defaults() {
        assert_eq!(replay_options(&argv(&["--trace", "t"])).unwrap(), (false, None));
        let (as_recorded, retry) = replay_options(&argv(&["--as-recorded", "--retry-engine-full"])).unwrap();
        assert!(as_recorded);
        assert_eq!(retry, Some(RetryPolicy::default()));
        let (_, retry) = replay_options(&argv(&[
            "--retry-engine-full", "--retry-max", "2", "--retry-max-wait", "0.5", "--retry-total-wait", "90",
        ]))
        .unwrap();
        assert_eq!(
            retry,
            Some(RetryPolicy {
                max_retries: 2,
                max_wait: std::time::Duration::from_millis(500),
                total_wait: std::time::Duration::from_secs(90),
            })
        );
    }

    #[test]
    fn a_retry_bound_without_the_retry_switch_or_with_a_bad_value_is_refused() {
        assert!(replay_options(&argv(&["--retry-max", "2"])).is_err());
        assert!(replay_options(&argv(&["--retry-max-wait", "5"])).is_err());
        assert!(replay_options(&argv(&["--retry-engine-full", "--retry-max", "x"])).is_err());
        assert!(replay_options(&argv(&["--retry-engine-full", "--retry-max-wait", "-1"])).is_err());
        assert!(replay_options(&argv(&["--retry-total-wait", "5"])).is_err());
        assert!(replay_options(&argv(&["--retry-engine-full", "--retry-total-wait", "x"])).is_err());
    }

    #[test]
    fn a_wait_too_big_for_a_duration_is_refused_not_a_panic() {
        // `Duration::from_secs_f64` panics on these.
        for huge in ["1e30", "1e300", "inf", "NaN"] {
            for flag in ["--retry-max-wait", "--retry-total-wait"] {
                let err = replay_options(&argv(&["--retry-engine-full", flag, huge])).expect_err(huge);
                assert!(err.contains(flag.trim_start_matches("--")), "{err}");
            }
        }
    }

    #[test]
    fn a_retry_max_wait_past_the_servers_retry_after_range_is_refused() {
        assert!(replay_options(&argv(&["--retry-engine-full", "--retry-max-wait", "60"])).is_ok());
        let err = replay_options(&argv(&["--retry-engine-full", "--retry-max-wait", "61"])).unwrap_err();
        assert!(err.contains("60"), "{err}");
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
    fn an_equivalence_side_is_a_live_endpoint_or_a_capture_never_both() {
        let args = argv(&["--spec-on", "http://127.0.0.1:8000", "--spec-off-capture", "off.json"]);
        assert_eq!(
            equivalence_side(&args, "spec-on"),
            Ok(EquivalenceSide::Endpoint("http://127.0.0.1:8000".into()))
        );
        assert_eq!(
            equivalence_side(&args, "spec-off"),
            Ok(EquivalenceSide::Capture("off.json".into()))
        );

        let both = argv(&["--spec-on", "http://x", "--spec-on-capture", "on.json"]);
        let err = equivalence_side(&both, "spec-on").expect_err("both given");
        assert!(err.contains("--spec-on") && err.contains("--spec-on-capture"), "{err}");

        let neither = argv(&["--spec-off", "http://x"]);
        let err = equivalence_side(&neither, "spec-on").expect_err("neither given");
        assert!(err.contains("--spec-on") && err.contains("--spec-on-capture"), "{err}");
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