//! Table test over `ignis-server`'s migrated diagnostic call sites (GitHub
//! #79): each one now emits a structured `ignis.<subsystem>.<event>` event
//! (JSON format, forced via `IGNIS_LOG_FORMAT=json`) instead of a bare
//! `eprintln!`. `ignis serve` keeps its whole event stream on stdout (no
//! severity-based stream split — that split is for one-shot commands like
//! `vendor-ninfer`), so these events land on stdout, not stderr.
//!
//! Coverage: 7 of the 14 migrated `main.rs` sites are exercised here —
//! every one reachable without a real `.ninfer` artifact fixture or a
//! `--features cuda` build. The other 7 are not testable from this
//! CPU-only file:
//! - `ignis.model.eos_missing`, `ignis.model.loaded`, `ignis.model.load_failed`
//!   live in `cuda_scheduler`, `#[cfg(feature = "cuda")]` — not even
//!   compiled into the default test binary (`docs/agents/testing.md`'s GPU
//!   profile: GPU-gated code stays out of the default `cargo test` run).
//! - `ignis.artifact.verified`, `ignis.artifact.load_failed`,
//!   `ignis.model.mock_compute` all require a real `.ninfer` artifact +
//!   sidecar past the checksum-verified loader — the same class of
//!   machine-local fixture dependency as `crates/artifact/tests/real_artifact.rs`,
//!   absent here.
//! - `ignis.config.thinking_invalid` needs a template whose
//!   `ThinkingCapabilities` reject the configured default; the only
//!   template reachable without an artifact (`SimpleTemplateProvider`)
//!   always reports `ThinkingCapabilities::permissive()`
//!   (`crates/server/src/template.rs`), so this path is unreachable without
//!   the same artifact fixture as above.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Run `ignis-server` to completion and return its last stdout line as a
/// logging JSON record. Only valid for paths that exit before binding a
/// socket (every "refusing to start" site, plus a bind failure).
fn run(args: &[&str]) -> serde_json::Value {
    run_with_envs(args, &[])
}

fn run_with_envs(args: &[&str], envs: &[(&str, &str)]) -> serde_json::Value {
    let exe = env!("CARGO_BIN_EXE_ignis-server");
    let mut command = Command::new(exe);
    command
        .args(args)
        .env("IGNIS_LOG_FORMAT", "json")
        .env("IGNIS_LOG_LEVEL", "info");
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.output().expect("run ignis-server");
    assert!(!output.status.success(), "expected a startup failure for {args:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_line = stdout
        .lines()
        .last()
        .unwrap_or_else(|| panic!("no stdout output for {args:?}"));
    serde_json::from_str(last_line)
        .unwrap_or_else(|error| panic!("stdout line is not a logging JSON record: {error}: {last_line:?}"))
}

/// Kills the child (and reaps it) when dropped — every `run_until` caller
/// reaches a site that never exits on its own (it goes on to bind a
/// socket), so the process must be torn down explicitly once its startup
/// events have been observed.
struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Spawn `ignis-server`, read stdout lines until one whose `event_name` is
/// in `until` appears (or a 10s timeout elapses), then kill the process.
/// Returns every JSON record observed, in order, up to and including the
/// match. `--bind 127.0.0.1:0` (an OS-assigned ephemeral port) is always
/// forced so these never collide with each other or with a real server.
fn run_until(args: &[&str], envs: &[(&str, &str)], until: &[&str]) -> Vec<serde_json::Value> {
    let exe = env!("CARGO_BIN_EXE_ignis-server");
    let mut command = Command::new(exe);
    command
        .args(args)
        .args(["--bind", "127.0.0.1:0"])
        .env("IGNIS_LOG_FORMAT", "json")
        .env("IGNIS_LOG_LEVEL", "info")
        .stdout(Stdio::piped());
    for (key, value) in envs {
        command.env(key, value);
    }
    let mut child = command.spawn().expect("spawn ignis-server");
    let stdout = child.stdout.take().expect("piped stdout");
    let guard = KillOnDrop(child);

    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let mut seen = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            panic!("timed out waiting for one of {until:?}; saw: {seen:?}");
        }
        let line = rx
            .recv_timeout(remaining)
            .unwrap_or_else(|_| panic!("timed out waiting for one of {until:?}; saw: {seen:?}"));
        let record: serde_json::Value = serde_json::from_str(&line)
            .unwrap_or_else(|error| panic!("stdout line is not a logging JSON record: {error}: {line:?}"));
        let matched = record["event_name"].as_str().is_some_and(|name| until.contains(&name));
        seen.push(record);
        if matched {
            break;
        }
    }
    drop(guard);
    seen
}

fn find<'a>(records: &'a [serde_json::Value], event_name: &str) -> &'a serde_json::Value {
    records
        .iter()
        .find(|record| record["event_name"] == event_name)
        .unwrap_or_else(|| panic!("no {event_name} record among {records:?}"))
}

#[test]
fn an_unrecognized_flag_emits_a_config_invalid_event() {
    let record = run(&["--this-flag-does-not-exist"]);
    assert_eq!(record["event_name"], "ignis.config.invalid");
    assert_eq!(record["severity_text"], "ERROR");
    assert!(
        record["attributes"]["error"].as_str().unwrap().contains("this-flag-does-not-exist"),
        "the offending flag should appear as a typed attribute, not only in body: {record}"
    );
}

#[test]
fn a_missing_artifact_sidecar_emits_a_sidecar_missing_event() {
    let missing = std::env::temp_dir().join("ignis-logging-test-does-not-exist.ninfer");
    let record = run(&["--artifact", missing.to_str().unwrap()]);
    assert_eq!(record["event_name"], "ignis.artifact.sidecar_missing");
    assert_eq!(record["severity_text"], "ERROR");
    assert_eq!(
        record["attributes"]["artifact"], missing.display().to_string(),
        "the artifact path should appear as a typed attribute, not only in body: {record}"
    );
}

#[test]
fn a_bind_conflict_emits_a_server_failed_event() {
    // Reserve a real ephemeral port and hold it open so ignis-server's own
    // bind attempt collides with it (the only CPU-only way to reach the
    // `server.serve` error path — the process exits(1) on failure, so this
    // uses `run`, not `run_until`).
    let held = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve a port");
    let addr = held.local_addr().expect("local_addr").to_string();
    let record = run(&["--bind", &addr]);
    assert_eq!(record["event_name"], "ignis.server.failed");
    assert_eq!(record["severity_text"], "ERROR");
    assert!(
        record["attributes"]["error"].as_str().is_some(),
        "the bind error should appear as a typed attribute: {record}"
    );
    drop(held);
}

#[test]
fn no_artifact_emits_placeholder_template_then_process_started() {
    let records = run_until(&[], &[], &["ignis.process.started"]);

    let placeholder = find(&records, "ignis.model.placeholder_template");
    assert_eq!(placeholder["severity_text"], "WARN");

    let started = find(&records, "ignis.process.started");
    assert_eq!(started["severity_text"], "INFO");
    assert!(
        started["attributes"]["model"].as_str().is_some(),
        "the loaded model should appear as a typed attribute: {started}"
    );
    assert!(
        started["attributes"]["bind"].as_str().is_some(),
        "the bind address should appear as a typed attribute: {started}"
    );
}

#[test]
fn a_valid_telemetry_path_emits_a_sink_selected_event() {
    let path = std::env::temp_dir().join(format!(
        "ignis-logging-test-telemetry-{}.jsonl",
        std::process::id()
    ));
    let records = run_until(
        &["--telemetry", path.to_str().unwrap()],
        &[],
        &["ignis.telemetry.sink_selected"],
    );
    let record = find(&records, "ignis.telemetry.sink_selected");
    assert_eq!(record["severity_text"], "INFO");
    assert_eq!(record["attributes"]["path"], path.display().to_string());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_unopenable_telemetry_path_emits_a_sink_failed_event_and_falls_back() {
    // A parent directory that does not exist: `FileSink::open` cannot
    // create the intermediate directory, so the open fails.
    let path = std::env::temp_dir()
        .join("ignis-logging-test-no-such-dir")
        .join("telemetry.jsonl");
    let records = run_until(
        &["--telemetry", path.to_str().unwrap()],
        &[],
        &["ignis.telemetry.sink_failed"],
    );
    let record = find(&records, "ignis.telemetry.sink_failed");
    assert_eq!(record["severity_text"], "WARN");
    assert_eq!(record["attributes"]["path"], path.display().to_string());
    assert!(
        record["attributes"]["error"].as_str().is_some(),
        "the open error should appear as a typed attribute: {record}"
    );
}
