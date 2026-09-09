//! One-shot-stdout-cleanliness (GitHub #79 Testing Decisions): running a
//! `vendor-ninfer` subcommand at a verbose logging level must not mix
//! diagnostic log lines into stdout — stdout stays reserved for command
//! results (`| jq .`/script-facing output), diagnostics go to stderr.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/vendor sits two levels below the repository root")
        .to_path_buf()
}

#[test]
fn verify_stdout_has_no_diagnostic_log_lines_at_trace_level() {
    let exe = env!("CARGO_BIN_EXE_vendor-ninfer");
    let manifest = repository_root().join("kernel/vendor/manifest.json");
    let output = Command::new(exe)
        .arg("verify")
        .arg("--manifest")
        .arg(&manifest)
        .env("IGNIS_LOG_LEVEL", "trace")
        .env("IGNIS_LOG_FORMAT", "json")
        .output()
        .expect("run vendor-ninfer");

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        assert!(
            serde_json::from_str::<serde_json::Value>(line).is_err(),
            "stdout must contain only command-result text, never a logging JSON record: {line:?}"
        );
    }
}
