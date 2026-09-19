//! Table test over `ignis-server`'s migrated diagnostic call sites (GitHub
//! #79): each one now emits a structured `ignis.<subsystem>.<event>` event
//! (JSON format, forced via `IGNIS_LOG_FORMAT=json`) instead of a bare
//! `eprintln!`. `ignis serve` keeps its whole event stream on stdout (no
//! severity-based stream split — that split is for one-shot commands like
//! `vendor-ninfer`), so these events land on stdout, not stderr.
//!
//! Coverage: 5 of the 12 migrated `main.rs` sites are exercised here —
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
fn an_artifact_path_that_is_not_there_emits_an_artifact_missing_event() {
    // GitHub #234: a named path that does not exist gets its own line. The
    // sidecar error would otherwise name a record next to a file that is not
    // there, and say nothing about the download that could have produced it.
    let missing = std::env::temp_dir().join("ignis-logging-test-does-not-exist.ninfer");
    let record = run(&["--artifact", missing.to_str().unwrap()]);
    assert_eq!(record["event_name"], "ignis.artifact.missing");
    assert_eq!(record["severity_text"], "ERROR");
    assert_eq!(
        record["attributes"]["artifact"], missing.display().to_string(),
        "the artifact path should appear as a typed attribute, not only in body: {record}"
    );
    assert!(
        record["body"].as_str().unwrap_or_default().contains("--model-download-path"),
        "the refusal should say how the model could be fetched instead: {record}"
    );
}

#[test]
fn a_missing_artifact_sidecar_emits_a_sidecar_missing_event() {
    // A file that is there but carries no provenance record (ADR 0002): the
    // load is refused before anything is read out of the container.
    let artifact = std::env::temp_dir().join("ignis-logging-test-no-sidecar.ninfer");
    std::fs::write(&artifact, b"not a real container").expect("write the stand-in artifact");
    let record = run(&["--artifact", artifact.to_str().unwrap()]);
    assert_eq!(record["event_name"], "ignis.artifact.sidecar_missing");
    assert_eq!(record["severity_text"], "ERROR");
    assert_eq!(
        record["attributes"]["artifact"], artifact.display().to_string(),
        "the artifact path should appear as a typed attribute, not only in body: {record}"
    );
    let _ = std::fs::remove_file(&artifact);
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
fn a_developer_policy_that_rerenders_history_warns_at_start() {
    // GitHub #209: joining or gathering developer messages loses prefix reuse
    // when one arrives mid-conversation, and the operator is told so once.
    for (policy, via_env) in [("into-system", false), ("after-system", true)] {
        let records = if via_env {
            run_until(&[], &[("IGNIS_DEVELOPER_MESSAGE_POLICY", policy)], &["ignis.process.started"])
        } else {
            run_until(&["--developer-message-policy", policy], &[], &["ignis.process.started"])
        };
        let warning = find(&records, "ignis.config.developer_policy_rerenders");
        assert_eq!(warning["severity_text"], "WARN");
        assert_eq!(warning["attributes"]["developer_message_policy"], policy);
    }
    for policy in ["inplace", "one-after-system", "reject"] {
        let records = run_until(&["--developer-message-policy", policy], &[], &["ignis.process.started"]);
        assert!(
            !records.iter().any(|r| r["event_name"] == "ignis.config.developer_policy_rerenders"),
            "{policy} moves no message and warns about nothing"
        );
    }
}

#[test]
fn no_artifact_emits_placeholder_template_then_process_started() {
    let records = run_until(&[], &[], &["ignis.process.started"]);

    let placeholder = find(&records, "ignis.model.placeholder_template");
    assert_eq!(placeholder["severity_text"], "WARN");
    // GitHub #234: which of the reasons it was. This binary is built without
    // `--features cuda`, so it never fetches weights it could not run.
    assert_eq!(placeholder["attributes"]["reason"], "not-supported");

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
    // GitHub #216: the pinned host arena is locked in RAM for the process's
    // whole life (ADR 0030), and no line anywhere used to say how large it
    // is. It is a typed number, not a rendered size string.
    assert!(
        started["attributes"]["kv_host_pool_bytes"].as_u64().is_some(),
        "the pinned KV-RAM arena's bytes should appear as a typed attribute: {started}"
    );
}

#[test]
fn the_startup_record_names_the_arena_the_operator_asked_for() {
    // The flag reaches the record, rather than the record naming a default
    // whatever was asked for (GitHub #216).
    let records = run_until(&["--kv-host-pool-bytes", "2G"], &[], &["ignis.process.started"]);
    let started = find(&records, "ignis.process.started");
    assert_eq!(started["attributes"]["kv_host_pool_bytes"], 2u64 << 30);
}

#[test]
fn the_retired_telemetry_flag_refuses_to_start() {
    // ADR 0025 removed the separate telemetry sink: a launch script still
    // passing `--telemetry` must fail by name, not start with its interval
    // counters silently gone somewhere else.
    let path = std::env::temp_dir().join("ignis-logging-test-telemetry.jsonl");
    let record = run(&["--telemetry", path.to_str().unwrap()]);
    assert_eq!(record["event_name"], "ignis.config.invalid");
    assert_eq!(record["severity_text"], "ERROR");
    assert!(
        record["attributes"]["error"].as_str().unwrap().contains("--telemetry"),
        "the retired flag should be named: {record}"
    );
}
