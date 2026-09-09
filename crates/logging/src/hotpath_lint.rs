//! A static/structural safeguard (GitHub #80, spec §40) against hot-path
//! logging regressions: a small grep-based scan over a fixed list of
//! prefill/decode/CUDA-graph-adjacent files, flagging any INFO-or-above
//! `tracing` call or raw `println!`/`eprintln!`/`print!` use — the shapes
//! spec §26/§40 both call out. This is a review-time safeguard, not a
//! runtime check: [`hot_path_files_have_no_violations`] (this module's own
//! test, run by plain `cargo test`) is what makes a regression "visible in
//! review" the way the issue asks, rather than only caught later by the G4
//! performance gate.
//!
//! `kernel/` (C++/CUDA) is out of scope (ADR on §33/34) — this only scans
//! Rust source.
//!
//! [`scan`] is pure (takes the file list and file contents as input, no
//! filesystem access of its own) so it is unit-testable against planted
//! fixtures without touching the real repo tree; [`scan_files_on_disk`] is
//! the thin, real-filesystem wrapper the crate's own test uses against the
//! actual hot-path file list.

use std::path::{Path, PathBuf};

/// The prefill/decode/CUDA-graph-adjacent Rust files this lint watches —
/// hand-maintained (spec §40 asks for review visibility, not perfect
/// automatic hot-path discovery): the per-token forward-pass runtime
/// (`ignis-core`'s scheduler/step/seq/layer modules), the safe step-ABI
/// runtime wrapper (`ignis-runtime`), and the server's own compute-adjacent
/// runtime glue. Deliberately excludes `main.rs`/`config.rs`/`api.rs`/
/// `telemetry.rs`/`loader.rs`/`template.rs`/`thinking.rs` — those are
/// startup/config/per-request-lifecycle code, exactly the granularity spec
/// §25 says INFO logging is for.
pub const HOT_PATH_FILES: &[&str] = &[
    "crates/core/src/scheduler.rs",
    "crates/core/src/concrete.rs",
    "crates/core/src/step.rs",
    "crates/core/src/seq.rs",
    "crates/core/src/kv.rs",
    "crates/core/src/gqa_layer.rs",
    "crates/core/src/gdn.rs",
    "crates/core/src/gdn_layer.rs",
    "crates/core/src/compute.rs",
    "crates/core/src/admission.rs",
    "crates/core/src/host.rs",
    "crates/core/src/prefix.rs",
    "crates/runtime/src/cuda_leaf.rs",
    "crates/server/src/runtime.rs",
    "crates/server/src/engine.rs",
    "crates/server/src/decoder.rs",
];

/// One flagged line: which file, which line number (1-based), and the
/// pattern that matched — enough for a reviewer (or a test) to locate and
/// judge it without re-running the scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub file: String,
    pub line_number: usize,
    pub line: String,
    pub pattern: &'static str,
}

/// Patterns that mark a line as a violation. `eprintln!` is checked before
/// `println!` deliberately: `"eprintln!".contains("println!")` is true (the
/// former is a superstring of the latter), so checking `println!` first
/// would mislabel every `eprintln!` line as a `println!` violation.
const VIOLATION_PATTERNS: &[&str] =
    &["tracing::info!", "tracing::warn!", "tracing::error!", "eprintln!", "println!", "print!"];

/// The suppression marker: a line (or the line immediately above it) that
/// contains this text is a reviewed, intentional exception — e.g. an ERROR
/// site that fires only on an already-exceptional failure path, not per
/// token/layer/kernel in the success case (spec §26 constrains *frequency*
/// in normal operation, not the mere existence of an error log). This
/// mirrors `#[allow(...)]`: it does not hide the call from a reader, only
/// from this automated scan, and it must carry a reason so the exemption
/// itself is reviewable.
const ALLOW_MARKER: &str = "hotpath-lint-allow:";

/// Comment markers that exempt a line from the scan — this lint is
/// deliberately grep-based (spec §40 explicitly allows "grep-based lint or a
/// small custom lint"), so a doc comment that merely *mentions* one of the
/// patterns (as this very file's own module doc does) must not self-flag.
fn is_commented_out(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("//!") || trimmed.starts_with('*')
}

/// Scan `contents` (the file's own source text) for [`VIOLATION_PATTERNS`],
/// labeling each finding with `file` for the caller's report. Pure — no
/// filesystem access — so tests can plant arbitrary fixture text.
pub fn scan(file: &str, contents: &str) -> Vec<Violation> {
    let lines: Vec<&str> = contents.lines().collect();
    let mut violations = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if is_commented_out(line) {
            continue;
        }
        let allowed = line.contains(ALLOW_MARKER)
            || idx.checked_sub(1).and_then(|prev| lines.get(prev)).is_some_and(|prev| prev.contains(ALLOW_MARKER));
        if allowed {
            continue;
        }
        for pattern in VIOLATION_PATTERNS {
            if line.contains(pattern) {
                violations.push(Violation {
                    file: file.to_owned(),
                    line_number: idx + 1,
                    line: line.trim().to_owned(),
                    pattern,
                });
                break; // one violation per line is enough to flag it
            }
        }
    }
    violations
}

/// [`scan`] over every file in `files`, read from `root` on disk. A file
/// listed but missing from disk is skipped, not an error — a hot-path file
/// getting renamed/removed is a repo-structure change this lint should not
/// itself block on; [`HOT_PATH_FILES`] staying accurate is a review
/// responsibility, not something this function can enforce.
pub fn scan_files_on_disk(root: &Path, files: &[&str]) -> Vec<Violation> {
    let mut violations = Vec::new();
    for file in files {
        let path: PathBuf = root.join(file);
        if let Ok(contents) = std::fs::read_to_string(&path) {
            violations.extend(scan(file, &contents));
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_planted_info_call_is_flagged() {
        let violations = scan(
            "fake/hot_path.rs",
            "fn decode_token() {\n    tracing::info!(\"one line per token\");\n}\n",
        );
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].line_number, 2);
        assert_eq!(violations[0].pattern, "tracing::info!");
    }

    #[test]
    fn a_planted_warn_and_error_call_are_both_flagged() {
        let violations = scan(
            "fake/hot_path.rs",
            "tracing::warn!(\"kv pressure\");\ntracing::error!(\"kernel launch failed\");\n",
        );
        assert_eq!(violations.len(), 2);
        assert_eq!(violations[0].pattern, "tracing::warn!");
        assert_eq!(violations[1].pattern, "tracing::error!");
    }

    #[test]
    fn a_planted_raw_println_is_flagged() {
        let violations = scan("fake/hot_path.rs", "println!(\"debugging in the hot loop\");\n");
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].pattern, "println!");
    }

    #[test]
    fn a_planted_raw_eprintln_is_flagged() {
        let violations = scan("fake/hot_path.rs", "eprintln!(\"oops\");\n");
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].pattern, "eprintln!");
    }

    #[test]
    fn debug_and_trace_calls_are_not_flagged() {
        let violations = scan(
            "fake/hot_path.rs",
            "tracing::debug!(\"per-layer detail\");\ntracing::trace!(\"per-token detail\");\n",
        );
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn a_comment_mentioning_the_pattern_is_not_flagged() {
        let violations = scan(
            "fake/hot_path.rs",
            "// do not add tracing::info! here, it fires per token\n\
             /// also not this doc comment mentioning println!\n",
        );
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn a_call_marked_allowed_on_its_own_line_is_not_flagged() {
        let violations = scan(
            "fake/hot_path.rs",
            "tracing::error!(\"leaf failed\"); // hotpath-lint-allow: failure path only, never per-token\n",
        );
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn a_call_marked_allowed_on_the_preceding_line_is_not_flagged() {
        let violations = scan(
            "fake/hot_path.rs",
            "// hotpath-lint-allow: failure path only, never per-token\ntracing::error!(\"leaf failed\");\n",
        );
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn an_allow_marker_does_not_exempt_an_unrelated_later_line() {
        let violations = scan(
            "fake/hot_path.rs",
            "// hotpath-lint-allow: this one call only\ntracing::error!(\"ok\");\ntracing::info!(\"not ok, no marker directly above this one\");\n",
        );
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].line_number, 3);
    }

    #[test]
    fn clean_code_produces_no_violations() {
        let violations = scan(
            "fake/hot_path.rs",
            "fn decode_token(&mut self) -> Token {\n    self.pool.step()\n}\n",
        );
        assert!(violations.is_empty());
    }

    #[test]
    fn scan_files_on_disk_skips_a_missing_file_instead_of_erroring() {
        let violations = scan_files_on_disk(Path::new("/does/not/exist"), &["nope.rs"]);
        assert!(violations.is_empty());
    }

    #[test]
    fn scan_files_on_disk_reads_and_flags_a_real_planted_file() {
        let dir = std::env::temp_dir().join(format!(
            "ignis-hotpath-lint-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("violating.rs"), "tracing::info!(\"per token\");\n").unwrap();
        std::fs::write(dir.join("clean.rs"), "fn noop() {}\n").unwrap();

        let violations = scan_files_on_disk(&dir, &["violating.rs", "clean.rs"]);

        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].file, "violating.rs");
    }

    /// The actual gate (spec §40): the repo's real hot-path files, scanned
    /// as they exist on disk right now, must have zero violations. This is
    /// what makes a regression visible under plain `cargo test` rather than
    /// only under the G4 performance gate — a future PR that adds an INFO
    /// call to `crates/core/src/step.rs`'s per-token path turns this test
    /// red immediately.
    #[test]
    fn hot_path_files_have_no_violations() {
        // `CARGO_MANIFEST_DIR` is `<repo>/crates/logging`; the workspace
        // root is two levels up.
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(Path::parent)
            .expect("crates/logging is two directories under the workspace root");

        let violations = scan_files_on_disk(repo_root, HOT_PATH_FILES);

        assert!(
            violations.is_empty(),
            "hot-path logging violation(s) found (spec §26/§40 — demote to DEBUG/TRACE or \
             convert to a state-transition event):\n{violations:#?}"
        );
    }
}
