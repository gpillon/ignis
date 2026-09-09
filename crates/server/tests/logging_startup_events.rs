//! Table test over `ignis-server`'s migrated startup-failure paths (GitHub
//! #79): each one now emits a structured `ignis.<subsystem>.<event>` event
//! (JSON format, forced via `IGNIS_LOG_FORMAT=json`) instead of a bare
//! `eprintln!`. `ignis serve` keeps its whole event stream on stdout (no
//! severity-based stream split — that split is for one-shot commands like
//! `vendor-ninfer`), so these events land on stdout, not stderr. Every case
//! here exits before the server ever binds a socket, so the subprocess is
//! fast and never blocks.

use std::process::Command;

fn run(args: &[&str]) -> serde_json::Value {
    let exe = env!("CARGO_BIN_EXE_ignis-server");
    let output = Command::new(exe)
        .args(args)
        .env("IGNIS_LOG_FORMAT", "json")
        .env("IGNIS_LOG_LEVEL", "info")
        .output()
        .expect("run ignis-server");
    assert!(!output.status.success(), "expected a startup failure for {args:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_line = stdout
        .lines()
        .last()
        .unwrap_or_else(|| panic!("no stdout output for {args:?}"));
    serde_json::from_str(last_line)
        .unwrap_or_else(|error| panic!("stdout line is not a logging JSON record: {error}: {last_line:?}"))
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
