//! Table test over `vendor-ninfer`'s 3 migrated diagnostic call sites
//! (GitHub #79): the usage error, the manifest-load failure, and the
//! generic command failure all now emit a structured `ignis.vendor.*`
//! event on stderr (JSON format, forced via `IGNIS_LOG_FORMAT=json`)
//! instead of a bare `eprintln!`. Command-result output stays on stdout
//! (see `stdout_cleanliness.rs`) — these events are asserted on stderr.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/vendor sits two levels below the repository root")
        .to_path_buf()
}

fn run(args: &[&str]) -> serde_json::Value {
    let exe = env!("CARGO_BIN_EXE_vendor-ninfer");
    let output = Command::new(exe)
        .args(args)
        .env("IGNIS_LOG_FORMAT", "json")
        .env("IGNIS_LOG_LEVEL", "info")
        .output()
        .expect("run vendor-ninfer");
    assert_eq!(
        output.status.code(),
        Some(2),
        "expected exit code 2 (usage/I/O error) for {args:?}: {output:?}"
    );
    // The usage-error path logs its event, then still prints the USAGE text
    // to stderr (so a human running it by hand sees the help) — so the
    // logging record is not necessarily the last stderr line; find the one
    // line that parses as a JSON object with an `event_name` field.
    let stderr = String::from_utf8_lossy(&output.stderr);
    stderr
        .lines()
        .find_map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).ok()?;
            value.get("event_name").is_some().then_some(value)
        })
        .unwrap_or_else(|| panic!("no logging JSON record on stderr for {args:?}: {stderr:?}"))
}

#[test]
fn an_unknown_flag_emits_a_usage_error_event() {
    let record = run(&["verify", "--not-a-real-flag"]);
    assert_eq!(record["event_name"], "ignis.vendor.usage_error");
    assert_eq!(record["severity_text"], "ERROR");
    assert!(
        record["attributes"]["error"].as_str().unwrap().contains("--not-a-real-flag"),
        "the offending flag should appear as a typed attribute, not only in body: {record}"
    );
}

#[test]
fn a_missing_manifest_emits_a_manifest_load_failed_event() {
    let missing = std::env::temp_dir().join("ignis-vendor-test-does-not-exist-manifest.json");
    let record = run(&["verify", "--manifest", missing.to_str().unwrap()]);
    assert_eq!(record["event_name"], "ignis.vendor.manifest_load_failed");
    assert_eq!(record["severity_text"], "ERROR");
    assert_eq!(
        record["attributes"]["manifest"], missing.display().to_string(),
        "the manifest path should appear as a typed attribute, not only in body: {record}"
    );
}

#[test]
fn an_unknown_command_emits_a_command_failed_event() {
    let manifest = repository_root().join("kernel/vendor/manifest.json");
    let record = run(&["not-a-real-command", "--manifest", manifest.to_str().unwrap()]);
    assert_eq!(record["event_name"], "ignis.vendor.command_failed");
    assert_eq!(record["severity_text"], "ERROR");
    assert_eq!(record["attributes"]["command"], "not-a-real-command");
    assert!(
        record["attributes"]["error"].as_str().is_some(),
        "the failure reason should appear as a typed attribute: {record}"
    );
}
